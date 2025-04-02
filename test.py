from mininet.net import Mininet
from mininet.topo import Topo
from mininet.log import setLogLevel, info
from mininet.util import quietRun
import time
import re
from topology import NetworkTopo, configure_network

class NetworkTester:
    def __init__(self):
        self.net = Mininet(topo=NetworkTopo())
        self.results = {}
        
    def setup(self):
        self.net.start()
        info("*** Starting network")
        configure_network(self.net)
        # Add minor delay for convergence
        time.sleep(1)  

    def run_tests(self):
        tests = [
            self.test_router_configuration,
            self.test_basic_connectivity,
            self.test_subnet_connectivity,
            self.test_performance_metrics,
            self.test_routing_paths,
            self.test_failure_scenarios
        ]
        
        for test in tests:
            try:
                test()
                info(f"{test.__name__} passed")
            except AssertionError as e:
                info(f"{test.__name__} failed: {str(e)}")
        
        return all(self.results.values())

    def test_router_configuration(self):
        """Verify router IPs, forwarding, and routing tables"""
        r1, r2 = self.net.get('r1', 'r2')
        
        # Verify IP forwarding
        assert '1' in r1.cmd('sysctl -n net.ipv4.ip_forward'), "R1 forwarding disabled"
        assert '1' in r2.cmd('sysctl -n net.ipv4.ip_forward'), "R2 forwarding disabled"
        
        # Verify interface IPs
        assert '192.168.1.1/24' in r1.cmd('ip addr show r1-eth0'), "R1 wrong LAN IP"
        assert '10.0.0.1/30' in r1.cmd('ip addr show r1-eth1'), "R1 wrong WAN IP"
        assert '192.168.2.1/24' in r2.cmd('ip addr show r2-eth1'), "R2 wrong LAN IP"
        assert '10.0.0.2/30' in r2.cmd('ip addr show r2-eth0'), "R2 wrong WAN IP"
        
        # Verify routing tables
        assert '192.168.2.0/24 via 10.0.0.2' in r1.cmd('ip route show'), "R1 missing route"
        assert '192.168.1.0/24 via 10.0.0.1' in r2.cmd('ip route show'), "R2 missing route"
        
        self.results['router_config'] = True

    def test_basic_connectivity(self):
        """Test basic ping between all adjacent nodes"""
        nodes = self.net.hosts + self.net.switches + [self.net.get('r1'), self.net.get('r2')]
        for src, dst in [(n1, n2) for n1 in nodes for n2 in nodes if n1 != n2]:
            if src in [self.net.get('r1'), self.net.get('r2')] and dst in [self.net.get('r1'), self.net.get('r2')]:
                continue  # Skip router-router test (tested elsewhere)
            result = src.cmd(f'ping -c1 -W1 {dst.IP()}')
            assert '1 received' in result, f"{src.name} -> {dst.name} failed"
        self.results['basic_connectivity'] = True

    def test_subnet_connectivity(self):
        """Validate cross-subnet communication"""
        h1, h3 = self.net.get('h1', 'h3')
        
        # Test basic connectivity
        assert '0% packet loss' in h1.cmd('ping -c2 -W1 h3'), "Cross-subnet ping failed"
        
        # Test traceroute path
        trace = h1.cmd('traceroute -n -m 5 h3')
        assert '10.0.0.1' in trace and '10.0.0.2' in trace, "Incorrect routing path"
        
        self.results['subnet_connectivity'] = True

    def test_performance_metrics(self):
        """Measure network performance characteristics"""
        h1, h3 = self.net.get('h1', 'h3')
        
        # Bandwidth test
        h1.cmd('iperf -s -p 5001 &')
        time.sleep(1)
        result = h3.cmd('iperf -c h1 -p 5001 -t 3')
        h1.cmd('killall iperf')
        
        # Parse bandwidth results
        bw = re.findall(r'(\d+\.\d+) Mbits/sec', result)
        assert float(bw[-1]) > 90, f"Low bandwidth: {bw[-1]} Mbps"
        
        # Latency test
        ping_result = h1.cmd('ping -c10 -i0.2 h3')
        times = re.findall(r'time=(\d+\.?\d*)', ping_result)
        avg_latency = sum(map(float, times))/len(times)
        assert avg_latency < 5, f"High latency: {avg_latency} ms"
        
        self.results['performance'] = True

    def test_routing_paths(self):
        """Verify correct routing behavior"""
        r1, r2 = self.net.get('r1', 'r2')
        
        # Test path from h1 to h3
        h1 = self.net.get('h1')
        route = h1.cmd('ip route get 192.168.2.10')
        assert 'via 192.168.1.1' in route, "H1 default route incorrect"
        
        # Test router-to-router connectivity
        assert '0% packet loss' in r1.cmd('ping -c2 10.0.0.2'), "Router interconnect failed"
        
        self.results['routing_paths'] = True

    def test_failure_scenarios(self):
        """Validate network resilience"""
        r1 = self.net.get('r1')
        
        # Simulate router failure
        r1.cmd('ifconfig r1-eth1 down')
        time.sleep(2)
        h1, h3 = self.net.get('h1', 'h3')
        result = h1.cmd('ping -c2 -W1 h3')
        assert '100% packet loss' in result, "Network didn't react to failure"
        
        # Restore connection
        r1.cmd('ifconfig r1-eth1 up')
        time.sleep(2)
        assert '0% packet loss' in h1.cmd('ping -c2 -W1 h3'), "Recovery failed"
        
        self.results['failure_resilience'] = True

    def cleanup(self):
        info("*** Stopping network")
        self.net.stop()

if __name__ == '__main__':
    setLogLevel('info')
    tester = NetworkTester()
    try:
        tester.setup()
        success = tester.run_tests()
        info("Final Test Results:")
        for test, result in tester.results.items():
            status = 'PASS' if result else 'FAIL'
            info(f"{test.ljust(20)} {status}")
        exit(0 if success else 1)
    finally:
        tester.cleanup()