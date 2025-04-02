from mininet.net import Mininet
from mininet.cli import CLI
from mininet.log import setLogLevel
import time
from topology import NetworkTopo, configure_network

def test_connectivity(net):
    print("\n[TEST] Basic Connectivity Check")
    h1, h2, h3, h4 = net.get('h1', 'h2', 'h3', 'h4')
    
    for src, dst in [(h1, h2), (h3, h4), (h1, h3), (h2, h4)]:
        result = src.cmd('ping -c 2 ' + dst.IP())
        if '0% packet loss' in result:
            print('✅ ' + src.name + ' can reach ' + dst.name)
        else:
            print('❌ ' + src.name + ' cannot reach ' + dst.name)

def test_routing(net):
    print("\n[TEST] Routing Check")
    h1, h3 = net.get('h1', 'h3')
    result = h1.cmd('ping -c 2 ' + h3.IP())
    if '0% packet loss' in result:
        print('✅ h1 can reach h3 via routers')
    else:
        print('❌ h1 cannot reach h3 via routers')

def test_latency(net):
    print("\n[TEST] Latency Measurement")
    h1, h3 = net.get('h1', 'h3')
    result = h1.cmd('ping -c 4 ' + h3.IP())
    print(result)

def test_bandwidth(net):
    print("\n[TEST] Bandwidth Test")
    h1, h3 = net.get('h1', 'h3')
    
    # Start iperf server on h3
    h3.cmd('iperf -s -D')
    time.sleep(2)
    
    # Run iperf client on h1
    result = h1.cmd('iperf -c ' + h3.IP() + ' -t 5')
    print(result)
    
    # Stop iperf server
    h3.cmd('pkill -f iperf')

def run_tests(net):
    test_connectivity(net)
    test_routing(net)
    test_latency(net)
    test_bandwidth(net)

def main():
    setLogLevel('info')
    net = Mininet(topo=NetworkTopo())
    net.start()
    configure_network(net)
    run_tests(net)
    CLI(net)
    net.stop()

if __name__ == '__main__':
    main()
