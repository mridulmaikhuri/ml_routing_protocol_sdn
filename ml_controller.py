#!/usr/bin/env python3
# ML-based Ryu Controller for Mininet Topology

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, DEAD_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet, ethernet, ether_types, ipv4, arp
from ryu.lib import hub
from ryu.topology import event, switches
from ryu.topology.api import get_switch, get_link
import networkx as nx
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from collections import defaultdict
import time
import pickle
import os

class MLController(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(MLController, self).__init__(*args, **kwargs)
        self.mac_to_port = {}
        self.ip_to_mac = {}
        self.mac_to_dpid = {}
        self.datapaths = {}
        self.topology_api_app = self
        self.net = nx.DiGraph()
        self.links = {}
        self.switches = {}
        self.hosts = {}
        self.gateway_macs = {}
        self.subnet_to_switch = {}
        self.model = None
        self.features_history = []
        self.link_stats = defaultdict(lambda: {'bytes': 0, 'packets': 0, 'last_bytes': 0, 'last_packets': 0})
        self.flow_stats = defaultdict(lambda: {'packet_count': 0, 'byte_count': 0, 'duration_sec': 0})

        # Configure virtual gateways for each subnet
        self.setup_virtual_gateways()

        # Start monitoring threads
        self.monitor_thread = hub.spawn(self._monitor)
        self.ml_thread = hub.spawn(self._ml_process)

        # Try to load pre-trained model if exists
        self.load_model()

    def setup_virtual_gateways(self):
        # Set up virtual gateway MAC and IP addresses for each subnet
        num_subnets = 5
        for i in range(1, num_subnets + 1):
            gateway_mac = '00:00:00:00:{:02x}:01'.format(i)
            gateway_ip = '10.0.{}.1'.format(i)
            self.gateway_macs[gateway_ip] = gateway_mac
            self.subnet_to_switch['10.0.{}.0/24'.format(i)] = i

    def load_model(self):
        try:
            if os.path.exists('ml_routing_model.pkl'):
                with open('ml_routing_model.pkl', 'rb') as f:
                    self.model = pickle.load(f)
                self.logger.info("Loaded pre-trained ML model")
            else:
                self.logger.info("No pre-trained model found, will train a new one")
                self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        except Exception as e:
            self.logger.error("Error loading model: {}".format(e))
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)

    def save_model(self):
        try:
            with open('ml_routing_model.pkl', 'wb') as f:
                pickle.dump(self.model, f)
            self.logger.info("Saved ML model")
        except Exception as e:
            self.logger.error("Error saving model: {}".format(e))
    
    @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def _state_change_handler(self, ev):
        datapath = ev.datapath
        if ev.state == MAIN_DISPATCHER:
            if datapath.id not in self.datapaths:
                self.logger.info('Register datapath: {:016x}'.format(datapath.id))
                self.datapaths[datapath.id] = datapath
        elif ev.state == DEAD_DISPATCHER:
            if datapath.id in self.datapaths:
                self.logger.info('Unregister datapath: {:016x}'.format(datapath.id))
                del self.datapaths[datapath.id]

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER,
                                          ofproto.OFPCML_NO_BUFFER)]
        self.add_flow(datapath, 0, match, actions)
        self.install_arp_handler(datapath)

    def install_arp_handler(self, datapath):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        match = parser.OFPMatch(eth_type=ether_types.ETH_TYPE_ARP)
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER,
                                          ofproto.OFPCML_NO_BUFFER)]
        self.add_flow(datapath, 1, match, actions)

    def add_flow(self, datapath, priority, match, actions, buffer_id=None, idle_timeout=0, hard_timeout=0):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
        if buffer_id:
            mod = parser.OFPFlowMod(datapath=datapath, buffer_id=buffer_id,
                                    priority=priority, match=match,
                                    instructions=inst, idle_timeout=idle_timeout,
                                    hard_timeout=hard_timeout)
        else:
            mod = parser.OFPFlowMod(datapath=datapath, priority=priority,
                                    match=match, instructions=inst,
                                    idle_timeout=idle_timeout, hard_timeout=hard_timeout)
        datapath.send_msg(mod)
    
    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in_handler(self, ev):
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        in_port = msg.match['in_port']
        dpid = datapath.id

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocols(ethernet.ethernet)[0]

        if eth.ethertype == ether_types.ETH_TYPE_LLDP:
            return
        
        dst_mac = eth.dst
        src_mac = eth.src

        self.mac_to_port.setdefault(dpid, {})
        self.mac_to_port[dpid][src_mac] = in_port
        self.mac_to_dpid[src_mac] = dpid

        if eth.ethertype == ether_types.ETH_TYPE_ARP:
            self.handle_arp(datapath, in_port, pkt)
            return

        if eth.ethertype == ether_types.ETH_TYPE_IP:
            ip_pkt = pkt.get_protocol(ipv4.ipv4)
            if ip_pkt:
                self.handle_ipv4(msg, datapath, in_port, eth, ip_pkt)
                return

        if dst_mac in self.mac_to_port[dpid]:
            out_port = self.mac_to_port[dpid][dst_mac]
        else:
            out_port = ofproto.OFPP_FLOOD

        actions = [parser.OFPActionOutput(out_port)]

        if out_port != ofproto.OFPP_FLOOD:
            match = parser.OFPMatch(in_port=in_port, eth_dst=dst_mac)
            if msg.buffer_id != ofproto.OFP_NO_BUFFER:
                self.add_flow(datapath, 1, match, actions, msg.buffer_id)
                return
            else:
                self.add_flow(datapath, 1, match, actions)

        data = None
        if msg.buffer_id == ofproto.OFP_NO_BUFFER:
            data = msg.data

        out = parser.OFPPacketOut(datapath=datapath, buffer_id=msg.buffer_id,
                                  in_port=in_port, actions=actions, data=data)
        datapath.send_msg(out)

    def handle_arp(self, datapath, in_port, pkt):
        arp_pkt = pkt.get_protocol(arp.arp)
        if not arp_pkt:
            return

        self.ip_to_mac[arp_pkt.src_ip] = arp_pkt.src_mac

        if arp_pkt.opcode == arp.ARP_REQUEST and arp_pkt.dst_ip in self.gateway_macs:
            self.reply_arp(datapath, in_port, pkt)
            return

        eth = pkt.get_protocols(ethernet.ethernet)[0]
        dst_mac = eth.dst
        src_mac = eth.src
        dpid = datapath.id

        self.mac_to_port.setdefault(dpid, {})
        self.mac_to_port[dpid][src_mac] = in_port

        self.flood_packet(datapath, in_port, pkt)

    def reply_arp(self, datapath, in_port, pkt):
        eth_pkt = pkt.get_protocols(ethernet.ethernet)[0]
        arp_pkt = pkt.get_protocol(arp.arp)

        gateway_mac = self.gateway_macs.get(arp_pkt.dst_ip)
        if not gateway_mac:
            return

        pkt_arp = packet.Packet()
        pkt_arp.add_protocol(ethernet.ethernet(
            ethertype=ether_types.ETH_TYPE_ARP,
            dst=eth_pkt.src,
            src=gateway_mac))
        pkt_arp.add_protocol(arp.arp(
            opcode=arp.ARP_REPLY,
            src_mac=gateway_mac,
            src_ip=arp_pkt.dst_ip,
            dst_mac=arp_pkt.src_mac,
            dst_ip=arp_pkt.src_ip))

        self.send_packet(datapath, in_port, pkt_arp)

    def flood_packet(self, datapath, in_port, pkt):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        data = None
        if isinstance(pkt, packet.Packet):
            data = pkt.data
        else:
            data = pkt

        actions = [parser.OFPActionOutput(ofproto.OFPP_FLOOD)]
        out = parser.OFPPacketOut(
            datapath=datapath,
            buffer_id=ofproto.OFP_NO_BUFFER,
            in_port=in_port,
            actions=actions,
            data=data)
        datapath.send_msg(out)

    def send_packet(self, datapath, port, pkt):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        pkt.serialize()
        data = pkt.data
        actions = [parser.OFPActionOutput(port=port)]
        out = parser.OFPPacketOut(
            datapath=datapath,
            buffer_id=ofproto.OFP_NO_BUFFER,
            in_port=ofproto.OFPP_CONTROLLER,
            actions=actions,
            data=data)
        datapath.send_msg(out)

    def handle_ipv4(self, msg, datapath, in_port, eth, ip_pkt):
        dpid = datapath.id
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        src_subnet = self.get_subnet(ip_pkt.src)
        dst_subnet = self.get_subnet(ip_pkt.dst)

        if src_subnet != dst_subnet:
            if dst_subnet in self.subnet_to_switch:
                dst_switch = self.subnet_to_switch[dst_subnet]
                path = self.get_optimal_path(dpid, dst_switch)

                if path and len(path) > 1:
                    self.install_path_flows(msg, path, eth, ip_pkt)
                    return

        if eth.dst in self.mac_to_port[dpid]:
            out_port = self.mac_to_port[dpid][eth.dst]
        else:
            out_port = ofproto.OFPP_FLOOD

        actions = [parser.OFPActionOutput(out_port)]

        if out_port != ofproto.OFPP_FLOOD:
            match = parser.OFPMatch(
                in_port=in_port,
                eth_type=ether_types.ETH_TYPE_IP,
                ipv4_src=ip_pkt.src,
                ipv4_dst=ip_pkt.dst)

            self.add_flow(datapath, 1, match, actions,
                          idle_timeout=60, hard_timeout=300)

        data = None
        if msg.buffer_id == ofproto.OFP_NO_BUFFER:
            data = msg.data

        out = parser.OFPPacketOut(
            datapath=datapath,
            buffer_id=msg.buffer_id,
            in_port=in_port,
            actions=actions,
            data=data)
        datapath.send_msg(out)

    def get_subnet(self, ip):
        octets = ip.split('.')
        if len(octets) == 4:
            return "10.0.{}.0/24".format(octets[2])
        return None

    def get_optimal_path(self, src_dpid, dst_dpid):
        if not self.net.has_node(src_dpid) or not self.net.has_node(dst_dpid):
            return None

        if not self.model or len(self.features_history) < 50:
            try:
                path = nx.shortest_path(self.net, src_dpid, dst_dpid)
                return path
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                return None

        try:
            paths = list(nx.all_simple_paths(self.net, src_dpid, dst_dpid, cutoff=5))

            if not paths:
                return None

            best_path = None
            best_score = float('inf')

            for path in paths:
                path_features = self.extract_path_features(path)
                predicted_delay = self.model.predict([path_features])[0]

                if predicted_delay < best_score:
                    best_score = predicted_delay
                    best_path = path

            return best_path

        except Exception as e:
            self.logger.error("Error in get_optimal_path: {}".format(e))
            try:
                return nx.shortest_path(self.net, src_dpid, dst_dpid)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                return None

    def extract_path_features(self, path):
        features = []
        features.append(len(path) - 1)

        total_bytes = 0
        total_packets = 0
        max_utilization = 0

        for i in range(len(path) - 1):
            src = path[i]
            dst = path[i + 1]
            link_key = (src, dst)

            if link_key in self.link_stats:
                stats = self.link_stats[link_key]
                total_bytes += stats['bytes']
                total_packets += stats['packets']

                bytes_rate = stats['bytes'] - stats['last_bytes']
                max_utilization = max(max_utilization, bytes_rate)

        features.append(total_bytes)
        features.append(total_packets)
        features.append(max_utilization)

        while len(features) < 10:
            features.append(0)

        return features[:10]

    def install_path_flows(self, msg, path, eth, ip_pkt):
        for i in range(len(path) - 1):
            curr_dpid = path[i]
            next_dpid = path[i + 1]

            out_port = self.get_port_for_next_hop(curr_dpid, next_dpid)
            if out_port is None:
                continue

            datapath = self.datapaths.get(curr_dpid)
            if datapath is None:
                continue

            parser = datapath.ofproto_parser

            match_forward = parser.OFPMatch(
                eth_type=ether_types.ETH_TYPE_IP,
                ipv4_src=ip_pkt.src,
                ipv4_dst=ip_pkt.dst)

            match_reverse = parser.OFPMatch(
                eth_type=ether_types.ETH_TYPE_IP,
                ipv4_src=ip_pkt.dst,
                ipv4_dst=ip_pkt.src)

            actions_forward = [parser.OFPActionOutput(out_port)]

            self.add_flow(datapath, 2, match_forward, actions_forward,
                          idle_timeout=60, hard_timeout=300)

            if i == len(path) - 2:
                dst_mac = eth.dst
                if dst_mac in self.mac_to_port.get(next_dpid, {}):
                    last_out_port = self.mac_to_port[next_dpid][dst_mac]
                    next_datapath = self.datapaths.get(next_dpid)

                    if next_datapath:
                        next_parser = next_datapath.ofproto_parser
                        next_actions = [next_parser.OFPActionOutput(last_out_port)]

                        self.add_flow(next_datapath, 2, match_forward, next_actions,
                                      idle_timeout=60, hard_timeout=300)

            if eth.src in self.mac_to_port.get(path[0], {}):
                first_in_port = self.mac_to_port[path[0]][eth.src]

                if i == 0:
                    src_datapath = self.datapaths.get(path[0])
                    if src_datapath:
                        src_parser = src_datapath.ofproto_parser

                        src_out_port = self.get_port_for_next_hop(path[0], path[1])
                        if src_out_port:
                            actions_reverse = [src_parser.OFPActionOutput(src_out_port)]
                            self.add_flow(src_datapath, 2, match_reverse, actions_reverse,
                                          idle_timeout=60, hard_timeout=300)

    def get_port_for_next_hop(self, src_dpid, dst_dpid):
        for link in self.links.values():
            if link['src'] == src_dpid and link['dst'] == dst_dpid:
                return link['src_port']
        return None
    
    @set_ev_cls(event.EventSwitchEnter)
    def switch_enter_handler(self, ev):
        switch = ev.switch
        dpid = switch.dp.id
        self.switches[dpid] = switch
        self.logger.info("Switch added: {}".format(dpid))
        self.update_topology()

    @set_ev_cls(event.EventSwitchLeave)
    def switch_leave_handler(self, ev):
        switch = ev.switch
        dpid = switch.dp.id
        if dpid in self.switches:
            del self.switches[dpid]
        self.logger.info("Switch removed: {}".format(dpid))
        self.update_topology()

    @set_ev_cls(event.EventLinkAdd)
    def link_add_handler(self, ev):
        link = ev.link
        src_dpid = link.src.dpid
        dst_dpid = link.dst.dpid
        src_port = link.src.port_no
        dst_port = link.dst.port_no

        link_id = "{}-{}".format(src_dpid, dst_dpid)
        self.links[link_id] = {
            'src': src_dpid,
            'dst': dst_dpid,
            'src_port': src_port,
            'dst_port': dst_port
        }

        self.logger.info("Link added: {}".format(link_id))
        self.update_topology()

    @set_ev_cls(event.EventLinkDelete)
    def link_delete_handler(self, ev):
        link = ev.link
        src_dpid = link.src.dpid
        dst_dpid = link.dst.dpid

        link_id = "{}-{}".format(src_dpid, dst_dpid)
        if link_id in self.links:
            del self.links[link_id]

        self.logger.info("Link removed: {}".format(link_id))
        self.update_topology()

    def update_topology(self):
        self.net.clear()

        for dpid in self.switches:
            self.net.add_node(dpid)

        for link_info in self.links.values():
            src = link_info['src']
            dst = link_info['dst']
            self.net.add_edge(src, dst, weight=1)

        self.logger.info("Topology updated: {} nodes, {} edges".format(len(self.net.nodes), len(self.net.edges)))


    def _monitor(self):
        while True:
            for dp in self.datapaths.values():
                self._request_stats(dp)
            hub.sleep(10)

    def _request_stats(self, datapath):
        self.logger.debug('send stats request: {:016x}'.format(datapath.id))
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        req = parser.OFPFlowStatsRequest(datapath)
        datapath.send_msg(req)

        req = parser.OFPPortStatsRequest(datapath, 0, ofproto.OFPP_ANY)
        datapath.send_msg(req)

    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def _flow_stats_reply_handler(self, ev):
        body = ev.msg.body
        dpid = ev.msg.datapath.id

        for stat in body:
            if 'ipv4_src' in stat.match and 'ipv4_dst' in stat.match:
                key = (dpid, stat.match['ipv4_src'], stat.match['ipv4_dst'])
                self.flow_stats[key] = {
                    'packet_count': stat.packet_count,
                    'byte_count': stat.byte_count,
                    'duration_sec': stat.duration_sec
                }

    @set_ev_cls(ofp_event.EventOFPPortStatsReply, MAIN_DISPATCHER)
    def _port_stats_reply_handler(self, ev):
        body = ev.msg.body
        dpid = ev.msg.datapath.id

        for stat in body:
            port_no = stat.port_no

            for link_id, link_info in self.links.items():
                if link_info['src'] == dpid and link_info['src_port'] == port_no:
                    dst_dpid = link_info['dst']
                    link_key = (dpid, dst_dpid)

                    self.link_stats[link_key]['last_bytes'] = self.link_stats[link_key]['bytes']
                    self.link_stats[link_key]['last_packets'] = self.link_stats[link_key]['packets']

                    self.link_stats[link_key]['bytes'] = stat.tx_bytes
                    self.link_stats[link_key]['packets'] = stat.tx_packets

                    bytes_rate = self.link_stats[link_key]['bytes'] - self.link_stats[link_key]['last_bytes']
                    if bytes_rate > 0:
                        weight = min(10, max(1, 1 + bytes_rate / 1000000))
                        if self.net.has_edge(dpid, dst_dpid):
                            self.net[dpid][dst_dpid]['weight'] = weight

    def _ml_process(self):
        while True:
            self._gather_training_data()

            if len(self.features_history) >= 50:
                self._train_model()

            hub.sleep(60)

    def _gather_training_data(self):
        for key, flow in self.flow_stats.items():
            if flow['packet_count'] > 0 and flow['duration_sec'] > 0:
                dpid, src_ip, dst_ip = key

                src_subnet = self.get_subnet(src_ip)
                dst_subnet = self.get_subnet(dst_ip)

                if src_subnet in self.subnet_to_switch and dst_subnet in self.subnet_to_switch:
                    src_switch = self.subnet_to_switch[src_subnet]
                    dst_switch = self.subnet_to_switch[dst_subnet]

                    try:
                        path = nx.shortest_path(self.net, src_switch, dst_switch)

                        features = self.extract_path_features(path)

                        if flow['byte_count'] > 0:
                            delay = flow['duration_sec'] * 1000 / (flow['byte_count'] / 1000)
                        else:
                            delay = flow['duration_sec'] * 1000

                        self.features_history.append((features, delay))

                        if len(self.features_history) > 1000:
                            self.features_history = self.features_history[-1000:]

                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        pass

    def _train_model(self):
        if len(self.features_history) < 50:
            return

        try:
            X = [features for features, _ in self.features_history]
            y = [delay for _, delay in self.features_history]

            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.model.fit(X, y)

            self.save_model()

            self.logger.info("ML model trained with {} samples".format(len(X)))

        except Exception as e:
            self.logger.error("Error training ML model: {}".format(e))

if __name__ == '__main__':
    import sys
    import os
    # Add the current directory to the Python path
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    from ryu.cmd import manager
    manager.main(['--ofp-tcp-listen-port', '6633', './ml_controller.py'])
