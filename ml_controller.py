from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, DEAD_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet, ethernet, ipv4, arp
from ryu.lib import hub
from ryu.topology import event
from ryu.topology.api import get_switch, get_link
from ryu.lib.packet import ether_types
import networkx as nx
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
import pickle
import os
import time
import logging
import csv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('MLController')

class MLController(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]
    GATEWAY_MAC_TEMPLATE = "00:00:00:00:{0:02x}:01"
    DEFAULT_BANDWIDTH = 100  # Mbps
    
    def __init__(self, *args, **kwargs):
        super(MLController, self).__init__(*args, **kwargs)
        self.topology_api_app = self
        self.net = nx.DiGraph()
        self.datapaths = {}
        self.mac_to_port = {}
        self.hosts = {}
        self.arp_table = {}
        self.flow_stats = {}
        self.port_stats = {}
        self.bandwidths = {}
        
        # ML Configuration
        self.model_file = 'ml_routing_model.pkl'
        self.data_file = 'network_data.csv'
        self.model = self._init_ml_model()
        self.training_data = []
        
        # Start background threads
        hub.spawn(self._monitor_network)
        hub.spawn(self._periodic_training)
        hub.spawn(self._save_data)

    def _init_ml_model(self):
        if os.path.exists(self.model_file):
            try:
                with open(self.model_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.error("Model load error: {0}".format(e))
        
        logger.info("Creating new RandomForest model")
        return RandomForestClassifier(n_estimators=100, random_state=42)

    def _monitor_network(self):
        while True:
            self._request_stats()
            hub.sleep(10)

    def _request_stats(self):
        for dp in self.datapaths.values():
            self._request_flow_stats(dp)
            self._request_port_stats(dp)

    def _request_flow_stats(self, datapath):
        ofp = datapath.ofproto
        parser = datapath.ofproto_parser
        req = parser.OFPFlowStatsRequest(datapath)
        datapath.send_msg(req)

    def _request_port_stats(self, datapath):
        ofp = datapath.ofproto
        parser = datapath.ofproto_parser
        req = parser.OFPPortStatsRequest(datapath, 0, ofp.OFPP_ANY)
        datapath.send_msg(req)

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in_handler(self, ev):
        msg = ev.msg
        datapath = msg.datapath
        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocol(ethernet.ethernet)
        
        if eth.ethertype == ether_types.ETH_TYPE_LLDP:
            return
        
        # Learn MAC addresses
        self.mac_to_port.setdefault(datapath.id, {})
        self.mac_to_port[datapath.id][eth.src] = msg.match['in_port']
        
        # Handle ARP
        if eth.ethertype == ether_types.ETH_TYPE_ARP:
            self._handle_arp(datapath, msg.in_port, pkt)
            return
            
        # IP Routing
        if eth.ethertype == ether_types.ETH_TYPE_IP:
            self._handle_ip(datapath, msg.in_port, pkt)

    def _handle_arp(self, datapath, in_port, pkt):
        arp_pkt = pkt.get_protocol(arp.arp)
        if arp_pkt.opcode == arp.ARP_REQUEST:
            self._handle_arp_request(datapath, in_port, arp_pkt)
        else:
            self._handle_arp_reply(pkt)

    def _handle_arp_reply(self, pkt):
        arp_pkt = pkt.get_protocol(arp.arp)
        if arp_pkt and arp_pkt.opcode == arp.ARP_REPLY:
            self.arp_table[arp_pkt.src_ip] = arp_pkt.src_mac
            logger.info("Learned ARP: {0} -> {1}".format(arp_pkt.src_ip, arp_pkt.src_mac))

    def _handle_arp_request(self, datapath, in_port, arp_pkt):
        if arp_pkt.dst_ip.startswith('10.0.') and arp_pkt.dst_ip.endswith('.1'):
            subnet = int(arp_pkt.dst_ip.split('.')[2])
            gateway_mac = self.GATEWAY_MAC_TEMPLATE.format(subnet)
            
            # Build ARP reply
            arp_reply = packet.Packet()
            arp_reply.add_protocol(ethernet.ethernet(
                ethertype=ether_types.ETH_TYPE_ARP,
                dst=arp_pkt.src_mac,
                src=gateway_mac))
            arp_reply.add_protocol(arp.arp(
                opcode=arp.ARP_REPLY,
                src_mac=gateway_mac,
                src_ip=arp_pkt.dst_ip,
                dst_mac=arp_pkt.src_mac,
                dst_ip=arp_pkt.src_ip))
            
            # Send packet out
            self._send_packet(datapath, in_port, arp_reply)
            logger.info("Sent ARP reply for {0} -> {1}".format(arp_pkt.dst_ip, gateway_mac))

    def _send_packet(self, datapath, in_port, pkt):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        pkt.serialize()
        
        actions = [parser.OFPActionOutput(ofproto.OFPP_FLOOD)]
        out = parser.OFPPacketOut(
            datapath=datapath,
            buffer_id=ofproto.OFP_NO_BUFFER,
            in_port=in_port,
            actions=actions,
            data=pkt.data
        )
        datapath.send_msg(out)

    def _handle_ip(self, datapath, in_port, pkt):
        ip_pkt = pkt.get_protocol(ipv4.ipv4)
        eth_pkt = pkt.get_protocol(ethernet.ethernet)
        
        src_subnet = '.'.join(ip_pkt.src.split('.')[:3])
        dst_subnet = '.'.join(ip_pkt.dst.split('.')[:3])
        
        if src_subnet != dst_subnet:
            self._handle_inter_subnet(datapath, in_port, eth_pkt, ip_pkt)
        else:
            self._handle_intra_subnet(datapath, in_port, eth_pkt, ip_pkt)

    def _handle_inter_subnet(self, datapath, in_port, eth_pkt, ip_pkt):
        try:
            path = self._get_ml_path(ip_pkt.src, ip_pkt.dst)
            if path:
                self._install_path_flows(path, ip_pkt.src, ip_pkt.dst)
        except Exception as e:
            logger.error("Routing error: {0}".format(e))

    def _get_ml_path(self, src_ip, dst_ip):
        try:
            paths = list(nx.all_simple_paths(self.net, src_ip, dst_ip))
            if not paths:
                return None
                
            features = [self._extract_features(path) for path in paths]
            predictions = self.model.predict_proba(features)
            return paths[np.argmax(predictions[:, 1])]
        except nx.NetworkXNoPath:
            return None
        except Exception as e:
            logger.error("Path finding error: {0}".format(e))
            return None

    def _extract_features(self, path):
        try:
            hop_count = len(path) - 1
            bandwidths = [self.bandwidths.get((path[i], path[i+1]), self.DEFAULT_BANDWIDTH)
                          for i in range(len(path)-1)]
            min_bandwidth = min(bandwidths)
            total_delay = hop_count * 5  # 5ms per hop approximation
            return [min_bandwidth, total_delay, hop_count]
        except Exception as e:
            logger.error("Feature extraction error: {0}".format(e))
            return [self.DEFAULT_BANDWIDTH, 0, 0]

    def _install_path_flows(self, path, src_ip, dst_ip):
        if len(path) < 2:
            return

        for i in range(len(path)-1):
            current_node = path[i]
            next_node = path[i+1]
            
            if not isinstance(current_node, int) or current_node not in self.datapaths:
                continue

            datapath = self.datapaths[current_node]
            parser = datapath.ofproto_parser
            
            try:
                out_port = self.net[current_node][next_node]['port']
            except KeyError:
                continue

            match = parser.OFPMatch(
                eth_type=ether_types.ETH_TYPE_IP,
                ipv4_src=src_ip,
                ipv4_dst=dst_ip
            )

            actions = [parser.OFPActionOutput(out_port)]
            if i == len(path)-2:
                dst_mac = self.arp_table.get(dst_ip)
                if dst_mac:
                    actions.insert(0, parser.OFPActionSetField(eth_dst=dst_mac))

            self.add_flow(
                datapath=datapath,
                priority=1000,
                match=match,
                actions=actions,
                idle_timeout=30,
                hard_timeout=60
            )
            logger.info("Installed flow on {0} for {1}->{2}".format(
                current_node, src_ip, dst_ip))

    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def _flow_stats_reply_handler(self, ev):
        body = ev.msg.body
        dpid = ev.msg.datapath.id
        current_time = time.time()

        for stat in body:
            if stat.match.get('ipv4_src') and stat.match.get('ipv4_dst'):
                key = (stat.match['ipv4_src'], stat.match['ipv4_dst'])
                
                if key in self.flow_stats:
                    prev = self.flow_stats[key]
                    time_diff = current_time - prev['timestamp']
                    byte_diff = stat.byte_count - prev['byte_count']
                    bandwidth = (byte_diff * 8) / (time_diff * 1e6) if time_diff > 0 else 0
                    self.training_data.append([
                        prev['bandwidth'],
                        prev['delay'],
                        prev['hop_count'],
                        bandwidth
                    ])
                
                self.flow_stats[key] = {
                    'byte_count': stat.byte_count,
                    'packet_count': stat.packet_count,
                    'timestamp': current_time,
                    'bandwidth': self.bandwidths.get(key, self.DEFAULT_BANDWIDTH),
                    'delay': len(self._get_ml_path(key[0], key[1])) * 5 if self._get_ml_path(key[0], key[1]) else 0,
                    'hop_count': len(self._get_ml_path(key[0], key[1])) if self._get_ml_path(key[0], key[1]) else 0
                }

    @set_ev_cls(ofp_event.EventOFPPortStatsReply, MAIN_DISPATCHER)
    def _port_stats_reply_handler(self, ev):
        body = ev.msg.body
        dpid = ev.msg.datapath.id
        current_time = time.time()

        for stat in body:
            port_no = stat.port_no
            for (src, dst), port in self.net.edges(data='port'):
                if src == dpid and port == port_no:
                    prev_stats = self.port_stats.get((src, dst), {})
                    if prev_stats:
                        time_diff = current_time - prev_stats['timestamp']
                        byte_diff = stat.tx_bytes - prev_stats['tx_bytes']
                        bw = (byte_diff * 8) / (time_diff * 1e6) if time_diff > 0 else 0
                        self.bandwidths[(src, dst)] = max(self.DEFAULT_BANDWIDTH - bw, 10)
                    
                    self.port_stats[(src, dst)] = {
                        'tx_bytes': stat.tx_bytes,
                        'rx_bytes': stat.rx_bytes,
                        'timestamp': current_time
                    }

    def _periodic_training(self):
        while True:
            if len(self.training_data) > 100:
                self._train_model()
            hub.sleep(300)

    def _train_model(self):
        try:
            X = np.array([d[:-1] for d in self.training_data])
            y = np.array([d[-1] for d in self.training_data])
            
            if len(X) == 0 or len(y) == 0:
                return
                
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            self.model.fit(X_train, y_train)
            
            with open(self.model_file, 'wb') as f:
                pickle.dump(self.model, f)
                
            logger.info("Model retrained with {0} samples".format(len(X)))
        except Exception as e:
            logger.error("Training error: {0}".format(e))

    def _save_data(self):
        while True:
            if self.training_data:
                try:
                    with open(self.data_file, 'a') as f:
                        writer = csv.writer(f)
                        writer.writerows(self.training_data)
                    self.training_data = []
                except Exception as e:
                    logger.error("Data save error: {0}".format(e))
            hub.sleep(60)

    @set_ev_cls(event.EventSwitchEnter)
    def switch_enter_handler(self, ev):
        switch = ev.switch
        self.net.add_node(switch.dp.id)
        self._discover_links()

    @set_ev_cls(event.EventSwitchLeave)
    def switch_leave_handler(self, ev):
        switch = ev.switch
        self.net.remove_node(switch.dp.id)
        self._discover_links()

    def _discover_links(self):
        links = get_link(self.topology_api_app, None)
        for link in links:
            src = link.src.dpid
            dst = link.dst.dpid
            self.net.add_edge(src, dst, port=link.src.port_no)
            self.net.add_edge(dst, src, port=link.dst.port_no)
            self.bandwidths[(src, dst)] = self.DEFAULT_BANDWIDTH
            self.bandwidths[(dst, src)] = self.DEFAULT_BANDWIDTH

    def add_flow(self, datapath, priority, match, actions, 
                buffer_id=None, idle_timeout=0, hard_timeout=0):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
        
        if buffer_id:
            mod = parser.OFPFlowMod(
                datapath=datapath,
                buffer_id=buffer_id,
                priority=priority,
                match=match,
                instructions=inst,
                idle_timeout=idle_timeout,
                hard_timeout=hard_timeout
            )
        else:
            mod = parser.OFPFlowMod(
                datapath=datapath,
                priority=priority,
                match=match,
                instructions=inst,
                idle_timeout=idle_timeout,
                hard_timeout=hard_timeout
            )
        datapath.send_msg(mod)

if __name__ == '__main__':
    from ryu.cmd import manager
    manager.main(['ml_controller.py', '--ofp-tcp-listen-port', '6633'])