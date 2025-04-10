#!/usr/bin/env python3

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, DEAD_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet, ethernet, ipv4, tcp, udp, ether_types
from ryu.lib import hub
from ryu.topology import event
from ryu.topology.api import get_switch, get_link
import networkx as nx
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle
import os
import time
import logging
import datetime

# Set up logging
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
        self.adjacency = {}
        self.bandwidths = {}
        self.flow_stats = {}

        # Create a graph for topology representation
        self.net = nx.DiGraph()
        
        # Initialize ML model
        self.model = self._initialize_ml_model()
        
        # Start background threads for monitoring
        self.monitor_thread = hub.spawn(self._monitor)
        self.path_update_thread = hub.spawn(self._path_update)
        
        logger.info("ML-based SDN Controller initialized")

    def _initialize_ml_model(self):
        """Initialize the ML model for path selection"""
        # Try to load a pre-trained model if it exists
        model_path = 'ml_routing_model.pkl'
        if os.path.exists(model_path):
            try:
                logger.info("Loading existing ML model...")
                with open(model_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.error(f"Error loading model: {e}")
        
        # Create a new model if no pre-trained model exists
        logger.info("Creating new ML model...")
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # We'll train this with some initial synthetic data
        # In a real implementation, you'd train this with actual network data
        X_synthetic = np.array([
            # [bandwidth, delay, packet_loss, hop_count]
            [95, 10, 0.1, 2],
            [80, 15, 0.3, 3],
            [60, 20, 0.5, 4],
            [40, 25, 0.7, 5],
            [90, 12, 0.2, 2],
            [75, 18, 0.4, 3],
            [50, 22, 0.6, 4],
            [30, 28, 0.8, 5],
        ])
        
        # Target is path preference (0: not preferred, 1: preferred)
        y_synthetic = np.array([1, 1, 0, 0, 1, 1, 0, 0])
        
        # Train the model with synthetic data
        model.fit(X_synthetic, y_synthetic)
        
        # Save the model
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        return model
    
    def _monitor(self):
        """Background thread for monitoring network statistics"""
        while True:
            for dp in self.datapaths.values():
                self._request_stats(dp)
            hub.sleep(10)
    
    def _path_update(self):
        """Background thread for updating path preferences based on ML model"""
        while True:
            if self.net.number_of_nodes() > 0:
                self._update_paths()
            hub.sleep(30)
    
    def _request_stats(self, datapath):
        """Request flow and port statistics from a datapath"""
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        
        # Request flow stats
        req = parser.OFPFlowStatsRequest(datapath)
        datapath.send_msg(req)
        
        # Request port stats
        req = parser.OFPPortStatsRequest(datapath, 0, ofproto.OFPP_ANY)
        datapath.send_msg(req)
    
    def _update_paths(self):
        """Update path preferences based on ML model predictions"""
        logger.info("Updating path preferences based on ML predictions")
        for src in self.net.nodes():
            for dst in self.net.nodes():
                if src != dst:
                    # Get all simple paths between src and dst
                    try:
                        paths = list(nx.all_simple_paths(self.net, src, dst))
                        if paths:
                            path_features = []
                            for path in paths:
                                # Calculate features for this path
                                features = self._calculate_path_features(path)
                                path_features.append((path, features))
                            
                            # Predict the best path using the ML model
                            best_path = self._predict_best_path(path_features)
                            
                            # Update flow rules for the best path
                            if best_path:
                                logger.info(f"Best path from {src} to {dst}: {best_path}")
                                # In a real implementation, you'd update flow rules here
                    except nx.NetworkXNoPath:
                        continue
    
    def _calculate_path_features(self, path):
        """Calculate features for a given path"""
        # In a real implementation, these would be actual network metrics
        hop_count = len(path) - 1
        
        # Initialize metrics
        bandwidth = 100  # Mbps
        delay = 10  # ms
        packet_loss = 0.1  # percentage
        
        # Calculate bandwidth (minimum bandwidth among all links in the path)
        for i in range(len(path) - 1):
            link = (path[i], path[i+1])
            if link in self.bandwidths:
                bandwidth = min(bandwidth, self.bandwidths[link])
        
        # Simulate delay and packet loss based on hop count
        delay += hop_count * 5
        packet_loss += hop_count * 0.1
        
        return [bandwidth, delay, packet_loss, hop_count]
    
    def _predict_best_path(self, path_features):
        """Use ML model to predict the best path"""
        if not path_features:
            return None
        
        # Extract features for prediction
        paths = [p[0] for p in path_features]
        features = np.array([p[1] for p in path_features])
        
        # Predict path preference
        predictions = self.model.predict_proba(features)
        
        # Get the index of the path with highest probability of being preferred
        best_idx = np.argmax(predictions[:, 1])
        
        return paths[best_idx]
    
    @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def _state_change_handler(self, ev):
        """Handle datapath state changes"""
        datapath = ev.datapath
        if ev.state == MAIN_DISPATCHER:
            if datapath.id not in self.datapaths:
                logger.info(f"Registered datapath: {datapath.id}")
                self.datapaths[datapath.id] = datapath
        elif ev.state == DEAD_DISPATCHER:
            if datapath.id in self.datapaths:
                logger.info(f"Unregistered datapath: {datapath.id}")
                del self.datapaths[datapath.id]
    
    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        """Handle switch features event"""
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        
        # Install table-miss flow entry
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER,
                                          ofproto.OFPCML_NO_BUFFER)]
        self.add_flow(datapath, 0, match, actions)
        
        logger.info(f"Switch {datapath.id} connected")
    
    def add_flow(self, datapath, priority, match, actions, buffer_id=None, idle_timeout=0, hard_timeout=0):
        """Add a flow entry to a datapath"""
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
    
    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in_handler(self, ev):
        """Handle packet in events"""
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        in_port = msg.match['in_port']
        
        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocols(ethernet.ethernet)[0]
        
        if eth.ethertype == ether_types.ETH_TYPE_LLDP:
            # Ignore LLDP packets
            return
        
        dst = eth.dst
        src = eth.src
        dpid = datapath.id
        
        # Learn MAC address to avoid flooding next time
        self.mac_to_port.setdefault(dpid, {})
        self.mac_to_port[dpid][src] = in_port
        
        # Update topology if needed
        if src not in self.net:
            self.net.add_node(src)
            self.net.add_edge(dpid, src, port=in_port)
            self.net.add_edge(src, dpid)
        
        # Check if we know the destination MAC
        if dst in self.mac_to_port[dpid]:
            out_port = self.mac_to_port[dpid][dst]
        else:
            out_port = ofproto.OFPP_FLOOD
        
        actions = [parser.OFPActionOutput(out_port)]
        
        # Install a flow to avoid packet_in next time
        if out_port != ofproto.OFPP_FLOOD:
            match = parser.OFPMatch(in_port=in_port, eth_dst=dst, eth_src=src)
            
            # IP packet with ML-based routing
            if eth.ethertype == ether_types.ETH_TYPE_IP:
                ip = pkt.get_protocol(ipv4.ipv4)
                if ip:
                    # Use ML-based routing for IP packets
                    src_ip = ip.src
                    dst_ip = ip.dst
                    
                    # Find the best path using our ML model
                    if src in self.net and dst in self.net:
                        try:
                            path = self._get_ml_path(src, dst)
                            if path and len(path) > 1:
                                # Update path in the network
                                self._install_path_flows(path, src_ip, dst_ip)
                                # Set the output port to the next hop
                                next_hop = path[1]
                                if next_hop in self.mac_to_port[dpid]:
                                    out_port = self.mac_to_port[dpid][next_hop]
                                    actions = [parser.OFPActionOutput(out_port)]
                        except Exception as e:
                            logger.error(f"Error finding ML path: {e}")
            
            # Add flow with a timeout
            if msg.buffer_id != ofproto.OFP_NO_BUFFER:
                self.add_flow(datapath, 1, match, actions, msg.buffer_id, idle_timeout=20)
            else:
                self.add_flow(datapath, 1, match, actions, idle_timeout=20)
        
        # Send packet out
        data = None
        if msg.buffer_id == ofproto.OFP_NO_BUFFER:
            data = msg.data
        
        out = parser.OFPPacketOut(datapath=datapath, buffer_id=msg.buffer_id,
                                  in_port=in_port, actions=actions, data=data)
        datapath.send_msg(out)
    
    def _get_ml_path(self, src, dst):
        """Get the ML-recommended path from src to dst"""
        if src == dst:
            return [src]
        
        try:
            # Get all simple paths between src and dst
            paths = list(nx.all_simple_paths(self.net, src, dst))
            if not paths:
                return None
            
            # Calculate features for each path
            path_features = []
            for path in paths:
                features = self._calculate_path_features(path)
                path_features.append((path, features))
            
            # Predict the best path using the ML model
            return self._predict_best_path(path_features)
        except Exception as e:
            logger.error(f"Error getting ML path: {e}")
            return None
    
    def _install_path_flows(self, path, src_ip, dst_ip):
        """Install flow rules along a path for specified IP addresses"""
        if len(path) < 2:
            return
        
        # Install flows for each switch in the path
        for i in range(len(path) - 1):
            if not isinstance(path[i], int):  # Skip non-switch nodes
                continue
                
            datapath = self.datapaths.get(path[i])
            if not datapath:
                continue
                
            # Get the output port for the next hop
            out_port = None
            for next_hop in self.mac_to_port.get(path[i], {}):
                if next_hop == path[i+1]:
                    out_port = self.mac_to_port[path[i]][next_hop]
                    break
            
            if not out_port:
                continue
                
            # Install flow rule
            parser = datapath.ofproto_parser
            match = parser.OFPMatch(eth_type=ether_types.ETH_TYPE_IP,
                                   ipv4_src=src_ip, ipv4_dst=dst_ip)
            actions = [parser.OFPActionOutput(out_port)]
            self.add_flow(datapath, 2, match, actions, idle_timeout=30)
    
    @set_ev_cls(event.EventSwitchEnter)
    def switch_enter_handler(self, ev):
        """Handle switch enter events"""
        switch = ev.switch
        dpid = switch.dp.id
        
        # Add switch to the graph
        if dpid not in self.net:
            self.net.add_node(dpid)
            self.switches.append(dpid)
            logger.info(f"Switch {dpid} added to topology")
        
        # Discover links
        self._discover_links()
    
    @set_ev_cls(event.EventSwitchLeave)
    def switch_leave_handler(self, ev):
        """Handle switch leave events"""
        switch = ev.switch
        dpid = switch.dp.id
        
        # Remove switch from the graph
        if dpid in self.net:
            self.net.remove_node(dpid)
            self.switches.remove(dpid)
            logger.info(f"Switch {dpid} removed from topology")
        
        # Update link information
        self._discover_links()
    
    def _discover_links(self):
        """Discover network links and update topology"""
        # Clear current links
        for node in list(self.net.nodes()):
            if node in self.switches:
                # Keep switch nodes but remove edges
                for edge in list(self.net.edges(node)):
                    self.net.remove_edge(edge[0], edge[1])
        
        # Get links and update the graph
        links_list = get_link(self.topology_api_app, None)
        if links_list:
            for link in links_list:
                src = link.src.dpid
                dst = link.dst.dpid
                src_port = link.src.port_no
                dst_port = link.dst.port_no
                
                # Add edges to the graph
                self.net.add_edge(src, dst, port=src_port)
                self.net.add_edge(dst, src, port=dst_port)
                
                # Initialize bandwidths
                self.bandwidths[(src, dst)] = 100  # Default 100 Mbps
                self.bandwidths[(dst, src)] = 100
                
                logger.info(f"Link added: {src}->{dst} via port {src_port}")
    
    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def _flow_stats_reply_handler(self, ev):
        """Handle flow statistics replies"""
        body = ev.msg.body
        dpid = ev.msg.datapath.id
        
        for stat in body:
            if 'ipv4_src' in stat.match and 'ipv4_dst' in stat.match:
                # Extract flow information
                src_ip = stat.match['ipv4_src']
                dst_ip = stat.match['ipv4_dst']
                
                # Update flow statistics
                key = (dpid, src_ip, dst_ip)
                self.flow_stats[key] = {
                    'byte_count': stat.byte_count,
                    'packet_count': stat.packet_count,
                    'duration_sec': stat.duration_sec,
                    'duration_nsec': stat.duration_nsec,
                    'timestamp': time.time()
                }
    
    @set_ev_cls(ofp_event.EventOFPPortStatsReply, MAIN_DISPATCHER)
    def _port_stats_reply_handler(self, ev):
        """Handle port statistics replies"""
        body = ev.msg.body
        dpid = ev.msg.datapath.id
        
        for stat in body:
            port_no = stat.port_no
            
            # Update port statistics
            # In a real implementation, you'd use these to update link metrics
            # such as bandwidth, delay, and packet loss
            
            # For demonstration, we'll update bandwidth based on port statistics
            # Find links using this port
            for src, dst, data in self.net.edges(data=True):
                if src == dpid and data.get('port') == port_no:
                    # Update bandwidth estimate
                    # This is a simplified example; in a real implementation,
                    # you'd calculate this based on byte counts and time intervals
                    self.bandwidths[(src, dst)] = 100 - (stat.tx_dropped * 0.1)
                    if self.bandwidths[(src, dst)] < 10:
                        self.bandwidths[(src, dst)] = 10
                    break

if __name__ == '__main__':
    from ryu.cmd import manager
    manager.main(['--ofp-tcp-listen-port', '6633', 'ml_controller'])