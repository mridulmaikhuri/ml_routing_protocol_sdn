from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
import os
import time
import datetime
import logging
import pickle
from collections import defaultdict, deque

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, DEAD_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet, ethernet, ipv4, tcp, udp, ether_types
from ryu.lib import hub
from ryu.topology import event
from ryu.topology.api import get_switch, get_link
from ryu.cmd import manager
import networkx as nx
import numpy as np

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('MLController')

class MLController(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(MLController, self).__init__(*args, **kwargs)
        self.topology_api_app = self
        self.mac_to_port = {}
        self.datapaths = {}
        self.switches = []
        self.hosts = {}
        self.multipath_group_ids = {}
        self.group_ids = []
        self.adjacency = defaultdict(dict)
        self.bandwidths = {}
        self.flow_stats = {}
        self.port_stats = {}
        self.path_history = {}
        self.flow_history = {}
        self.path_performance_history = []
        
        self.net = nx.DiGraph()
        
        self.model, self.feature_scaler = self._initialize_ml_model()
        
        # Start monitoring threads
        self.monitor_thread = hub.spawn(self._monitor)
        self.path_update_thread = hub.spawn(self._path_update)
        self.model_update_thread = hub.spawn(self._periodic_model_update)
        self.anomaly_detection_thread = hub.spawn(self._periodic_anomaly_detection)
        
        logger.info("Python 2 Compatible ML Ensemble-based SDN Controller initialized")

    # initialize ml model
    def _initialize_ml_model(self):
        model_path = 'ensemble_routing_model.pkl'
        scaler_path = 'feature_scaler.pkl'
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            try:
                logger.info("Loading existing ensemble ML model...")
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
                return model, scaler
            except Exception as e:
                logger.error("Error loading model: %s", str(e))
        
        logger.info("Creating new ensemble ML model...")
        
        # Create synthetic training data
        X_synthetic = np.array([
            # Bandwidth delay loss hops 
            [95, 10, 0.1, 2],   
            [80, 15, 0.3, 3],    
            [60, 20, 0.5, 4],   
            [40, 25, 0.7, 5],    
            [90, 12, 0.2, 2],    
            [75, 18, 0.4, 3],    
            [50, 22, 0.6, 4],    
            [30, 28, 0.8, 5]     
        ])
        
        y_synthetic = np.array([1, 1, 0, 0, 1, 1, 0, 0])
        
        # Create feature scaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_synthetic)
        
        # Initialize component models
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        svm_model = SVC(probability=True, kernel=b'rbf', random_state=42)
        gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        
        # Create voting ensemble model
        ensemble_model = VotingClassifier(
            estimators=[
                ('rf', rf_model),
                ('svm', svm_model),
                ('gb', gb_model)
            ],
            voting='soft'
        )
        
        # Train the ensemble model
        ensemble_model.fit(X_scaled, y_synthetic)
        
        # Save models using protocol=2 for Python 2 compatibility
        with open(model_path, 'wb') as f:
            pickle.dump(ensemble_model, f, protocol=2)
        
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f, protocol=2)
        
        return ensemble_model, scaler

    # Monitor network stats
    def _monitor(self):
        while True:
            for dp in self.datapaths.values():
                self._request_stats(dp)
            hub.sleep(10)

    # Request stats from switch
    def _request_stats(self, datapath):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        
        req = parser.OFPFlowStatsRequest(datapath)
        datapath.send_msg(req)
        
        req = parser.OFPPortStatsRequest(datapath, 0, ofproto.OFPP_ANY)
        datapath.send_msg(req)

    # triggers when a new switch connects to the network or disconnects to the network to register or unregister a switch
    @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def _state_change_handler(self, ev):
        datapath = ev.datapath
        if ev.state == MAIN_DISPATCHER:
            if datapath.id not in self.datapaths:
                logger.info("Registered datapath: %016x", datapath.id)
                self.datapaths[datapath.id] = datapath
        elif ev.state == DEAD_DISPATCHER:
            if datapath.id in self.datapaths:
                logger.info("Unregistered datapath: %016x", datapath.id)
                del self.datapaths[datapath.id]

    # triggers when a new switch connects to install default flow rule to send unmatched packet to controller
    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER,
                                        ofproto.OFPCML_NO_BUFFER)]
        self.add_flow(datapath, 0, match, actions)
        
        logger.info("Switch %016x connected", datapath.id)

    # used to install flow rules on the switch using controller
    def add_flow(self, datapath, priority, match, actions, buffer_id=None, idle_timeout=0, hard_timeout=0):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS,
                                           actions)]
        if buffer_id:
            mod = parser.OFPFlowMod(datapath=datapath, buffer_id=buffer_id,
                                  priority=priority, match=match,
                                  instructions=inst, idle_timeout=idle_timeout,
                                  hard_timeout=hard_timeout)
        else:
            mod = parser.OFPFlowMod(datapath=datapath, priority=priority,
                                  match=match, instructions=inst,
                                  idle_timeout=idle_timeout,
                                  hard_timeout=hard_timeout)
        datapath.send_msg(mod)

    # handles packet  which are coming in switch
    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in_handler(self, ev):
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        in_port = msg.match['in_port']

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocols(ethernet.ethernet)[0]

        # Ignore LLDP packets
        if eth.ethertype == ether_types.ETH_TYPE_LLDP:
            return

        dst = eth.dst
        src = eth.src
        dpid = datapath.id

        # Learn the source MAC address for the switch
        self.mac_to_port.setdefault(dpid, {})
        self.mac_to_port[dpid][src] = in_port

        # Add the source host to the network graph if not already present
        if src not in self.net:
            self.net.add_node(src)
            self.net.add_edge(dpid, src, port=in_port)
            self.net.add_edge(src, dpid)

        # Default action: flood if destination unknown
        out_port = ofproto.OFPP_FLOOD
        if dst in self.mac_to_port[dpid]:
            out_port = self.mac_to_port[dpid][dst]

        use_ml_path = False
        src_ip = None
        dst_ip = None

        # If the packet is IPv4, attempt to use ML-based path decision
        if eth.ethertype == ether_types.ETH_TYPE_IP:
            ip_pkt = pkt.get_protocol(ipv4.ipv4)
            if ip_pkt:
                src_ip = ip_pkt.src
                dst_ip = ip_pkt.dst
                
                # Identify traffic type
                traffic_type = self._traffic_classification(pkt)
                
                # Proceed only if both source and destination exist in our network graph
                if src in self.net and dst in self.net:
                    try:
                        path = self._get_ml_path(src, dst, traffic_type)
                        if path and len(path) > 1:
                            # Install flow entries along the ML-selected path
                            self._install_path_flows(path, src_ip, dst_ip, traffic_type)
                            next_hop = path[1]
                            if next_hop in self.mac_to_port[dpid]:
                                out_port = self.mac_to_port[dpid][next_hop]
                                use_ml_path = True
                                
                                # Record this path selection for future feedback
                                path_key = "%s-%s" % (src_ip, dst_ip)
                                self.path_history[path_key] = {
                                    'path': path,
                                    'features': self._calculate_path_features(path),
                                    'timestamp': time.time(),
                                    'traffic_type': traffic_type
                                }
                    except Exception as e:
                        logger.error("Error finding ML path: %s", str(e))

        # Set the action list with the final out_port
        actions = [parser.OFPActionOutput(out_port)]

        # Only install a flow if the packet is not being flooded
        if out_port != ofproto.OFPP_FLOOD:
            if use_ml_path and eth.ethertype == ether_types.ETH_TYPE_IP:
                # More specific match for IP traffic when ML path is used
                match = parser.OFPMatch(
                    in_port=in_port,
                    eth_type=ether_types.ETH_TYPE_IP,
                    ipv4_src=src_ip,
                    ipv4_dst=dst_ip
                )
                # Higher priority for ML-based decisions
                priority = 2
            else:
                # Fallback: match on Ethernet source and destination
                match = parser.OFPMatch(in_port=in_port, eth_src=src, eth_dst=dst)
                priority = 1

            if msg.buffer_id != ofproto.OFP_NO_BUFFER:
                self.add_flow(datapath, priority, match, actions, msg.buffer_id, idle_timeout=20)
            else:
                self.add_flow(datapath, priority, match, actions, idle_timeout=20)

        # Send packet out
        data = None
        if msg.buffer_id == ofproto.OFP_NO_BUFFER:
            data = msg.data

        out = parser.OFPPacketOut(
            datapath=datapath,
            buffer_id=msg.buffer_id,
            in_port=in_port,
            actions=actions,
            data=data
        )
        datapath.send_msg(out)

    # used to classify traffic type
    def _traffic_classification(self, pkt):
        eth = pkt.get_protocols(ethernet.ethernet)[0]
        traffic_type = "default"
        
        if eth.ethertype == ether_types.ETH_TYPE_IP:
            ip_pkt = pkt.get_protocol(ipv4.ipv4)
            if ip_pkt:
                protocol = ip_pkt.proto
                
                if protocol == 6:  # TCP
                    tcp_pkt = pkt.get_protocol(tcp.tcp)
                    if tcp_pkt:
                        # Web traffic
                        if tcp_pkt.dst_port == 80 or tcp_pkt.dst_port == 443:
                            traffic_type = "web"
                        # SSH
                        elif tcp_pkt.dst_port == 22:
                            traffic_type = "interactive"
                        # Email
                        elif tcp_pkt.dst_port == 25 or tcp_pkt.dst_port == 587:
                            traffic_type = "email"
                
                elif protocol == 17:  # UDP
                    udp_pkt = pkt.get_protocol(udp.udp)
                    if udp_pkt:
                        # DNS
                        if udp_pkt.dst_port == 53:
                            traffic_type = "dns"
                        # VoIP or video
                        elif udp_pkt.dst_port >= 16384 and udp_pkt.dst_port <= 32767:
                            traffic_type = "realtime"
        
        return traffic_type

    # get best path based on ML model with traffic type consideration
    def _get_ml_path(self, src, dst, traffic_type="default"):
        if src == dst:
            return [src]
        
        try:
            # Find all simple paths between src and dst
            paths = list(nx.all_simple_paths(self.net, src, dst, cutoff=10))
            if not paths:
                return None
            
            path_features = []
            for path in paths:
                # Calculate base features
                features = self._calculate_path_features(path)
                
                # Adjust features based on traffic type
                if traffic_type == "web":
                    # Web traffic: prioritize low latency
                    features[1] *= 0.9  
                elif traffic_type == "interactive":
                    # Interactive traffic: prioritize stability
                    features[2] *= 0.8  
                elif traffic_type == "email":
                    # Email traffic: prioritize bandwidth
                    features[0] *= 1.1 
                
                path_features.append((path, features))
            
            # Use ensemble model to predict best path
            return self._predict_best_path(path_features)
        except Exception as e:
            logger.error("Error getting ML path: %s", str(e))
            return None

    # calculate path features for ML model input
    def _calculate_path_features(self, path):
        hop_count = len(path) - 1
        
        # Default values
        bandwidth = 100  
        delay = 10     
        packet_loss = 0.1  
        
        # Calculate actual metrics based on observed network stats
        for i in xrange(len(path) - 1):
            if isinstance(path[i], (int, long)) and isinstance(path[i+1], (int, long)):  
                link = (path[i], path[i+1])
                if link in self.bandwidths:
                    bandwidth = min(bandwidth, self.bandwidths[link])
                
                # Calculate delay based on link properties
                link_delay = 5 
                if link in self.flow_stats:
                    if 'delay' in self.flow_stats[link]:
                        link_delay = self.flow_stats[link]['delay']
                    
                delay += link_delay
                
                # Calculate packet loss based on link properties
                link_loss = 0.05  
                if link in self.flow_stats:
                    if 'packet_loss' in self.flow_stats[link]:
                        link_loss = self.flow_stats[link]['packet_loss']
                
                packet_loss += link_loss
        
        return [bandwidth, delay, packet_loss, hop_count]

    # Used to predict best path 
    def _predict_best_path(self, path_features):
        if not path_features:
            return None
        
        paths = [p[0] for p in path_features]
        features = np.array([p[1] for p in path_features])
        
        # Scale features before prediction
        features_scaled = self.feature_scaler.transform(features)
        
        # Get prediction probabilities from ensemble model
        predictions = self.model.predict_proba(features_scaled)
        
        # Select path with highest probability of being "good"
        best_idx = np.argmax(predictions[:, 1])
        
        # Also consider path diversity for load balancing
        good_path_indices = np.where(predictions[:, 1] > max(0.7, predictions[best_idx, 1] - 0.1))[0]
        if len(good_path_indices) > 1:
            # Calculate selection probabilities based on prediction confidence
            probs = predictions[good_path_indices, 1]
            probs = probs / np.sum(probs)  
            
            # Select path probabilistically
            selected_idx = np.random.choice(good_path_indices, p=probs)
            return paths[selected_idx]
        
        return paths[best_idx]

    # install flow entries along the selected path
    def _install_path_flows(self, path, src_ip, dst_ip, traffic_type="default"):
        if len(path) < 2:
            return
        
        # Set appropriate timeout and priority based on traffic type
        idle_timeout = 30  
        hard_timeout = 60 
        priority = 2     
        
        if traffic_type == "web":
            idle_timeout = 20  # Web flows might be shorter
            priority = 3       # Higher priority for web traffic
        elif traffic_type == "interactive":
            idle_timeout = 120  # Interactive sessions tend to be longer
            hard_timeout = 300
            priority = 3
        
        for i in xrange(len(path) - 1): 
            if not isinstance(path[i], (int, long)):
                continue
                
            datapath = self.datapaths.get(path[i])
            if not datapath:
                continue
                
            out_port = None
            for next_hop in self.mac_to_port.get(path[i], {}):
                if next_hop == path[i+1]:
                    out_port = self.mac_to_port[path[i]][next_hop]
                    break
            
            if not out_port:
                continue
                
            parser = datapath.ofproto_parser
            match = parser.OFPMatch(eth_type=ether_types.ETH_TYPE_IP,
                                   ipv4_src=src_ip, ipv4_dst=dst_ip)
            actions = [parser.OFPActionOutput(out_port)]
            self.add_flow(datapath, priority, match, actions, 
                          idle_timeout=idle_timeout, hard_timeout=hard_timeout)

    # event which gets trigerred when a new switch enters the network
    @set_ev_cls(event.EventSwitchEnter)
    def switch_enter_handler(self, ev):
        """Handle switch connection event"""
        switch = ev.switch
        dpid = switch.dp.id
        
        if dpid not in self.net:
            self.net.add_node(dpid)
            self.switches.append(dpid)
            logger.info("Switch %016x added to topology", dpid)
        
        self._discover_links()

    # discover network topology links
    def _discover_links(self):
        # Remove existing links from switches
        for node in self.net.nodes():
            if node in self.switches:
                for edge in self.net.edges(node):
                    self.net.remove_edge(edge[0], edge[1])
        
        # Discover new links
        links_list = get_link(self.topology_api_app, None)
        if links_list:
            for link in links_list:
                src = link.src.dpid
                dst = link.dst.dpid
                src_port = link.src.port_no
                dst_port = link.dst.port_no
                
                self.net.add_edge(src, dst, port=src_port)
                self.net.add_edge(dst, src, port=dst_port)
                
                # Initialize link bandwidths
                self.bandwidths[(src, dst)] = 100
                self.bandwidths[(dst, src)] = 100
                
                logger.info("Link added: %016x->%016x via port %d", src, dst, src_port)

    # handle flow statistics event
    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def _flow_stats_reply_handler(self, ev):
        body = ev.msg.body
        dpid = ev.msg.datapath.id
        
        for stat in body:
            if 'ipv4_src' in stat.match and 'ipv4_dst' in stat.match:
                src_ip = stat.match['ipv4_src']
                dst_ip = stat.match['ipv4_dst']
                
                key = (dpid, src_ip, dst_ip)
                previous_stats = self.flow_stats.get(key, None)
                
                current_stats = {
                    'byte_count': stat.byte_count,
                    'packet_count': stat.packet_count,
                    'duration_sec': stat.duration_sec,
                    'duration_nsec': stat.duration_nsec,
                    'timestamp': time.time()
                }
                
                # Calculate throughput if we have previous measurements
                if previous_stats:
                    time_diff = current_stats['timestamp'] - previous_stats['timestamp']
                    if time_diff > 0:
                        byte_diff = current_stats['byte_count'] - previous_stats['byte_count']
                        throughput_bps = (byte_diff * 8) / time_diff 
                        throughput_mbps = throughput_bps / 1000000  
                        
                        current_stats['throughput'] = throughput_mbps
                        
                        # Update path history with throughput data for feedback
                        path_key = "%s-%s" % (src_ip, dst_ip)
                        if path_key in self.path_history:
                            self.path_history[path_key]['throughput'] = throughput_mbps
                
                self.flow_stats[key] = current_stats

    # handles port statistics reply event
    @set_ev_cls(ofp_event.EventOFPPortStatsReply, MAIN_DISPATCHER)
    def _port_stats_reply_handler(self, ev):
        body = ev.msg.body
        dpid = ev.msg.datapath.id
        
        for stat in body:
            port_no = stat.port_no
            
            # Update link metrics based on port statistics
            for src, dst, data in self.net.edges(data=True):
                if src == dpid and data.get('port') == port_no:
                    # Calculate bandwidth utilization
                    if port_no in self.port_stats.get(dpid, {}):
                        prev_stat = self.port_stats[dpid][port_no]
                        time_diff = time.time() - prev_stat['timestamp']
                        if time_diff > 0:
                            # Calculate throughput
                            byte_diff = stat.tx_bytes - prev_stat['tx_bytes']
                            bit_rate = (byte_diff * 8) / time_diff  
                            
                            capacity = 1000000000  
                            utilization = (bit_rate / capacity) * 100
                            
                            # Update available bandwidth (100 - utilization %)
                            self.bandwidths[(src, dst)] = max(5, 100 - utilization)
                            
                            # Calculate packet loss rate
                            tx_packets_diff = stat.tx_packets - prev_stat['tx_packets']
                            tx_errors_diff = stat.tx_errors - prev_stat['tx_errors']
                            if tx_packets_diff > 0:
                                packet_loss_rate = tx_errors_diff / tx_packets_diff
                                
                                # Update link metrics for flow decisions
                                link = (src, dst)
                                if link not in self.flow_stats:
                                    self.flow_stats[link] = {}
                                
                                self.flow_stats[link]['packet_loss'] = packet_loss_rate
                                
                                # Update path history with loss data for feedback
                                for path_key, path_data in self.path_history.items():
                                    if link in zip(path_data['path'][:-1], path_data['path'][1:]):
                                        self.path_history[path_key]['packet_loss'] = packet_loss_rate
                    
                    # Store current stats for future calculations
                    if dpid not in self.port_stats:
                        self.port_stats[dpid] = {}
                    
                    self.port_stats[dpid][port_no] = {
                        'tx_bytes': stat.tx_bytes,
                        'rx_bytes': stat.rx_bytes,
                        'tx_packets': stat.tx_packets,
                        'rx_packets': stat.rx_packets,
                        'tx_errors': stat.tx_errors,
                        'rx_errors': stat.rx_errors,
                        'timestamp': time.time()
                    }
                    
                    break
    
    # used to update paths periodically
    def _path_update(self):
        while True:
            if self.net.number_of_nodes() > 0:
                self._update_paths()
            hub.sleep(30)

    # updates path preference based on ML model predictions
    def _update_paths(self):
        logger.info("Updating path preferences based on ML ensemble predictions")
        for src in self.net.nodes():
            for dst in self.net.nodes():
                if src != dst:
                    try:
                        paths = list(nx.all_simple_paths(self.net, src, dst, cutoff=10))
                        if paths:
                            path_features = []
                            for path in paths:
                                features = self._calculate_path_features(path)
                                path_features.append((path, features))
                            
                            best_path = self._predict_best_path(path_features)
                            
                            if best_path:
                                logger.info("Best path from %s to %s: %s", src, dst, best_path)
                                
                                # Store the path selection for future feedback
                                path_key = "%s-%s" % (src, dst)
                                self.path_history[path_key] = {
                                    'path': best_path,
                                    'features': self._calculate_path_features(best_path),
                                    'timestamp': time.time(),
                                    'throughput': 0,
                                    'delay': 0,
                                    'packet_loss': 0
                                }
                    except nx.NetworkXNoPath:
                        continue
                    except Exception as e:
                        logger.error("Error updating path from %s to %s: %s", src, dst, str(e))

    # updates the model periodically
    def _periodic_model_update(self):
        while True:
            if len(self.path_history) > 10:
                self._update_model_with_feedback()
            hub.sleep(300)  

    # updates the model with historical path performance data
    def _update_model_with_feedback(self):
        try:
            current_time = time.time()
            X_update = []
            y_update = []
            
            for path_key, path_data in self.path_history.items():
                if current_time - path_data['timestamp'] >= 120:
                    features = path_data['features']
                    
                    # Calculate performance metrics from flow stats
                    throughput = path_data.get('throughput', 0)
                    delay = path_data.get('delay', 100)
                    packet_loss = path_data.get('packet_loss', 1.0)
                    
                    # Determine if this was a "good" path selection
                    performance_score = throughput/10 - delay/10 - packet_loss*10
                    is_good_path = 1 if performance_score > 0 else 0
                    
                    X_update.append(features)
                    y_update.append(is_good_path)
                    
                    # Keep only recent history
                    if current_time - path_data['timestamp'] > 600:  
                        del self.path_history[path_key]
            
            if len(X_update) > 5:  # Only update if we have enough new samples
                logger.info("Updating model with %d new samples", len(X_update))
                X_update = np.array(X_update)
                y_update = np.array(y_update)
                
                # Scale features
                X_scaled = self.feature_scaler.transform(X_update)
                
                # Update the model with new data
                for name, estimator in self.model.named_estimators_.items():
                    if hasattr(estimator, 'partial_fit'):
                        try:
                            estimator.partial_fit(X_scaled, y_update, classes=[0, 1])
                        except:
                            pass  
        
        except Exception as e:
            logger.error("Error updating model: %s", str(e))

    # periodically runs anomaly detection
    def _periodic_anomaly_detection(self):
        while True:
            self._detect_anomalies()
            hub.sleep(60)  # Run every minute

    # detect network anomalies based on traffic patterns
    def _detect_anomalies(self):
        try:
            # Calculate flow statistics
            flow_rates = {}
            current_time = time.time()
            
            for key, stats in self.flow_stats.items():
                if isinstance(key, tuple) and len(key) == 3: 
                    dpid, src_ip, dst_ip = key
                    
                    # Skip old records
                    if 'timestamp' not in stats or current_time - stats['timestamp'] > 60:
                        continue
                    
                    # Calculate flow rate
                    if 'prev_byte_count' in stats:
                        time_diff = stats['timestamp'] - stats['prev_timestamp']
                        byte_diff = stats['byte_count'] - stats['prev_byte_count']
                        
                        if time_diff > 0:
                            flow_rate = byte_diff / time_diff  # bytes per second
                            flow_rates[key] = flow_rate
                    
                    # Store current values for next calculation
                    stats['prev_byte_count'] = stats['byte_count']
                    stats['prev_timestamp'] = stats['timestamp']
            
            # Detect sudden changes in flow rates
            for key, rate in flow_rates.items():
                dpid, src_ip, dst_ip = key
                
                # Get historical rates
                history_key = "%s-%s" % (src_ip, dst_ip)
                if history_key in self.flow_history:
                    avg_rate = self.flow_history[history_key]['avg_rate']
                    
                    # Check for anomalies (sudden increase or decrease)
                    if rate > avg_rate * 3:  
                        logger.warning("Traffic surge detected from %s to %s: %.2f KB/s", 
                                      src_ip, dst_ip, rate/1024)
                        
                        # Update anomaly database
                        if 'anomalies' not in self.flow_history[history_key]:
                            self.flow_history[history_key]['anomalies'] = []
                        
                        self.flow_history[history_key]['anomalies'].append({
                            'timestamp': current_time,
                            'type': 'surge',
                            'rate': rate
                        })
                        
                        # Take action if needed (e.g., reroute, rate limit)
                        if rate > avg_rate * 10:  # Severe surge
                            self._mitigate_traffic_surge(src_ip, dst_ip)
                    
                    # Update average rate (moving average)
                    self.flow_history[history_key]['avg_rate'] = 0.8 * avg_rate + 0.2 * rate
                else:
                    # Initialize history
                    self.flow_history[history_key] = {
                        'avg_rate': rate,
                        'timestamps': [current_time],
                        'rates': [rate]
                    }
        
        except Exception as e:
            logger.error("Error detecting anomalies: %s", str(e))

    # apply mitigation strategies for traffic surges
    def _mitigate_traffic_surge(self, src_ip, dst_ip):
        try:
            # Implement rate limiting or rerouting
            for dp_id, datapath in self.datapaths.items():
                parser = datapath.ofproto_parser
                ofproto = datapath.ofproto
                
                # Match the surge traffic
                match = parser.OFPMatch(
                    eth_type=ether_types.ETH_TYPE_IP,
                    ipv4_src=src_ip,
                    ipv4_dst=dst_ip
                )
                
                # Apply rate limiting or reroute
                if hasattr(parser, 'OFPMeterMod'):
                    # Configure meter
                    meter_id = hash((src_ip, dst_ip)) % 1000 + 1
                    bands = [
                        parser.OFPMeterBandDrop(rate=1000, burst_size=100)
                    ]
                    req = parser.OFPMeterMod(
                        datapath=datapath,
                        command=ofproto.OFPMC_ADD,
                        flags=ofproto.OFPMF_KBPS,
                        meter_id=meter_id,
                        bands=bands
                    )
                    datapath.send_msg(req)
                    
                    # Apply meter to flow
                    actions = [parser.OFPActionOutput(ofproto.OFPP_NORMAL)]
                    inst = [
                        parser.OFPInstructionMeter(meter_id),
                        parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)
                    ]
                    
                    mod = parser.OFPFlowMod(
                        datapath=datapath,
                        priority=100,
                        match=match,
                        instructions=inst,
                        idle_timeout=120
                    )
                    datapath.send_msg(mod)
                    
                    logger.info("Applied rate limiting to traffic from %s to %s", src_ip, dst_ip)
        
        except Exception as e:
            logger.error("Error mitigating traffic surge: %s", str(e))

if __name__ == '__main__':
    manager.main(['--ofp-tcp-listen-port', '6633', 'ml_controller'])