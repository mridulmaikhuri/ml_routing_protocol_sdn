import unittest
import subprocess
import time
import re
from mininet.topo import Topo
from mininet.net import Mininet
from mininet.node import OVSController
from mininet.log import setLogLevel, info
from mininet.util import dumpNodeConnections
from topology import ComplexLoopTopo

class TestComplexTopology(unittest.TestCase):
    """Test suite for the complex loop topology"""
    
    @classmethod
    def setUpClass(cls):
        """Start up the topology once for all tests"""
        setLogLevel('info')
        cls.topo = ComplexLoopTopo()
        cls.net = Mininet(topo=cls.topo, controller=OVSController)
        cls.net.start()
        
        # Enable STP on all switches to handle loops
        for switch in cls.net.switches:
            switch.cmd('ovs-vsctl set bridge ' + switch.name + ' stp_enable=true')
        
        # Wait for STP to converge (typically takes a few seconds)
        info('*** Waiting for STP to converge...\n')
        time.sleep(15)
        
    @classmethod
    def tearDownClass(cls):
        """Stop the network after all tests are done"""
        cls.net.stop()
    
    def test_node_connectivity(self):
        """Test basic connectivity between all nodes"""
        info('*** Testing basic connectivity between all hosts\n')
        hosts = self.net.hosts
        
        # Test ping between all host pairs
        dropped = 0
        total = 0
        for src in hosts:
            for dst in hosts:
                if src != dst:
                    total += 1
                    result = src.cmd('ping -c1 -W1 %s' % dst.IP())
                    if '0% packet loss' not in result:
                        dropped += 1
                        info('Ping from %s to %s failed\n' % (src.name, dst.name))
        
        success_rate = ((total - dropped) / total) * 100
        info('Connectivity test: %d/%d successful pings (%.1f%%)\n' % 
             (total - dropped, total, success_rate))
        
        # We expect a high success rate (allow for some STP convergence issues)
        self.assertGreater(success_rate, 95, "Connectivity test failed with too many dropped pings")
    
    def test_path_redundancy(self):
        """Test that path redundancy works when links fail"""
        info('*** Testing path redundancy with link failures\n')
        
        # Choose two distant hosts
        h1 = self.net.get('h1')
        h10 = self.net.get('h10')
        
        # Verify they can communicate
        result = h1.cmd('ping -c2 -W1 %s' % h10.IP())
        self.assertIn('0% packet loss', result, "Hosts could not communicate before link failure")
        
        # Find a path between them using traceroute
        traceroute = h1.cmd('traceroute -n %s' % h10.IP())
        info('Path before failure: %s\n' % traceroute)
        
        # Break the primary link
        s1 = self.net.get('s1')
        s2 = self.net.get('s2')
        info('Breaking link s1-s2\n')
        self.net.configLinkStatus('s1', 's2', 'down')
        
        # Wait for STP to reconverge
        info('Waiting for reconvergence...\n')
        time.sleep(10)
        
        # Verify they can still communicate
        result = h1.cmd('ping -c2 -W2 %s' % h10.IP())
        self.assertIn('0% packet loss', result, "Hosts could not communicate after link failure")
        
        # Find the new path
        new_traceroute = h1.cmd('traceroute -n %s' % h10.IP())
        info('Path after failure: %s\n' % new_traceroute)
        
        # Restore the link
        info('Restoring link s1-s2\n')
        self.net.configLinkStatus('s1', 's2', 'up')
        time.sleep(5)
    
    def test_bandwidth(self):
        """Test bandwidth on different links using iperf"""
        info('*** Testing bandwidth on different links\n')
        
        # Test on a high-bandwidth path
        h2 = self.net.get('h2')
        h8 = self.net.get('h8')
        
        # Start iperf server on h8
        h8.cmd('iperf -s &')
        time.sleep(1)
        
        # Run iperf client on h2
        result = h2.cmd('iperf -c %s -t 5' % h8.IP())
        info('Bandwidth h2->h8: %s\n' % result)
        
        # Extract bandwidth value using regex
        bandwidth_match = re.search(r'(\d+(\.\d+)?) Mbits/sec', result)
        if bandwidth_match:
            bandwidth = float(bandwidth_match.group(1))
            info('Measured bandwidth: %.2f Mbits/sec\n' % bandwidth)
            # We expect reasonable bandwidth (varies by system)
            self.assertGreater(bandwidth, 1, "Bandwidth test failed with too low throughput")
        
        # Stop iperf server
        h8.cmd('kill %iperf')
    
    def test_latency(self):
        """Test latency on different links"""
        info('*** Testing latency on different links\n')
        
        # Test latency between hosts on different segments
        host_pairs = [
            ('h1', 'h5'),  # Regular link
            ('h3', 'h6'),  # Link with 10ms delay
            ('h4', 'h7')   # Link with 15ms delay
        ]
        
        for src_name, dst_name in host_pairs:
            src = self.net.get(src_name)
            dst = self.net.get(dst_name)
            
            # Run ping test
            result = src.cmd('ping -c 5 %s' % dst.IP())
            
            # Extract average latency
            latency_match = re.search(r'min/avg/max/mdev = [\d.]+/([\d.]+)', result)
            if latency_match:
                latency = float(latency_match.group(1))
                info('Latency from %s to %s: %.2f ms\n' % (src_name, dst_name, latency))
    
    def test_stp_convergence(self):
        """Test that STP properly handles loops"""
        info('*** Testing STP convergence\n')
        
        for switch in self.net.switches:
            # Check STP status
            result = switch.cmd('ovs-vsctl get bridge %s stp_enable' % switch.name)
            self.assertEqual(result.strip(), 'true', "STP not enabled on %s" % switch.name)
            
            # Check port states
            port_states = switch.cmd('ovs-ofctl show %s | grep addr' % switch.name)
            info('%s port states: %s\n' % (switch.name, port_states))
            
            # Verify at least one port is not in blocking state (state 1)
            # This is a basic check that STP is actually working
            self.assertIn('state=FORWARD', port_states, 
                         "No forwarding ports found on %s, STP may not be working" % switch.name)

if __name__ == '__main__':
    # Run tests
    unittest.main()