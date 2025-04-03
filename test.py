from mininet.net import Mininet
from mininet.util import dumpNodeConnections
from mininet.log import setLogLevel
from topology import ComplexTopology
import time

def run_tests():
    # Setup network
    print("Starting network setup...")
    setLogLevel('info')
    topo = ComplexTopology()
    net = Mininet(topo=topo)
    net.start()

    # Get hosts
    hosts = net.hosts
    h1, h2, h3, h4, h5, h6 = hosts[:6]

    # Test 1: Basic connectivity
    print("\n=== Test 1: Basic Connectivity ===")
    dumpNodeConnections(net.hosts)
    connectivity_result = net.pingAll()
    print(f"Ping all result: {connectivity_result}% packet loss")

    # Test 2: Bandwidth test using iperf
    print("\n=== Test 2: Bandwidth Tests ===")
    for src, dst in [(h1, h6), (h2, h5), (h3, h4)]:
        net.iperf((src, dst), seconds=5)
        print(f"Bandwidth test {src} -> {dst} complete")

    # Test 3: Path redundancy test
    print("\n=== Test 3: Path Redundancy Test ===")
    # Disable one link and test connectivity
    net.configLinkStatus('s1', 's2', 'down')
    print("Disabled link between s1 and s2")
    redundancy_result = net.pingAll()
    print(f"Ping all with s1-s2 down: {redundancy_result}% packet loss")
    net.configLinkStatus('s1', 's2', 'up')
    print("Restored link between s1 and s2")

    # Test 4: Latency test
    print("\n=== Test 4: Latency Test ===")
    for src, dst in [(h1, h6), (h2, h5)]:
        result = net.ping([src, dst], timeout=5)
        print(f"Latency {src} -> {dst}: {result}% packet loss")

    # Test 5: Link failure and recovery
    print("\n=== Test 5: Link Failure and Recovery ===")
    print("Testing with multiple link failures...")
    net.configLinkStatus('s5', 's6', 'down')
    net.configLinkStatus('s4', 's7', 'down')
    failure_result = net.pingAll()
    print(f"Ping all with multiple links down: {failure_result}% packet loss")
    
    # Restore links and verify recovery
    net.configLinkStatus('s5', 's6', 'up')
    net.configLinkStatus('s4', 's7', 'up')
    recovery_result = net.pingAll()
    print(f"Ping all after recovery: {recovery_result}% packet loss")

    # Test 6: Throughput under load
    print("\n=== Test 6: Throughput Under Load ===")
    server = h6
    clients = [h1, h2, h3]
    server.cmd('iperf -s &')
    time.sleep(1)
    for client in clients:
        result = client.cmd(f'iperf -c {server.IP()} -t 5')
        print(f"Throughput {client} -> {server}: {result}")

    # Cleanup
    print("\nCleaning up...")
    net.stop()

def main():
    print("Starting comprehensive network tests...")
    print("=" * 50)
    run_tests()
    print("=" * 50)
    print("Tests completed!")

if __name__ == '__main__':
    main()