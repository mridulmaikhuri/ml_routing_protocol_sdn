from mininet.net import Mininet
from mininet.cli import CLI
from mininet.log import setLogLevel
import time
from topology import topology  # Importing your topology

def test_connectivity(net):
    print("\n[TEST] Basic Connectivity Check")
    for h in range(1, 11):
        for h_target in range(h+1, 11):
            src = net.get('h' + str(h))
            dst = net.get('h' + str(h_target))
            result = src.cmd('ping -c 2 ' + dst.IP())
            if '0% packet loss' in result:
                print(src.name + ' can reach ' + dst.name + ' ✅')
            else:
                print(src.name + ' cannot reach ' + dst.name + ' ❌')

def test_routing(net):
    print("\n[TEST] Routing Check")
    for r in range(1, 6):
        router = net.get('r' + str(r))
        result = router.cmd('ip route')
        print('Routing table for ' + router.name + ':\n' + result)

def test_latency(net):
    print("\n[TEST] Latency Measurement")
    h1, h2 = net.get('h1', 'h10')
    result = h1.cmd('ping -c 4 ' + h2.IP())
    print(result)

def test_bandwidth(net):
    print("\n[TEST] Bandwidth Test")
    h1, h10 = net.get('h1', 'h10')
    h10.cmd('iperf -s -D')
    time.sleep(2)
    result = h1.cmd('iperf -c ' + h10.IP() + ' -t 5')
    print(result)
    h10.cmd('pkill -f iperf')

def run_tests(net):
    test_connectivity(net)
    test_routing(net)
    test_latency(net)
    test_bandwidth(net)

def main():
    setLogLevel('info')
    net = Mininet()
    topology()  # Call your topology function
    run_tests(net)
    net.stop()

if __name__ == '__main__':
    main()
