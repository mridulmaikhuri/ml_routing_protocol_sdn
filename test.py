from mininet.log import setLogLevel, info
from mininet.util import dumpNodeConnections
import time
from topology import topology
import traceback

def test_connectivity(net):
    info("*** Testing connectivity between all hosts\n")
    host_names = ['h{0}'.format(i + 1) for i in range(10)]
    hosts = [net.get(h) for h in host_names]
    total_tests = 0
    successful_tests = 0
    
    for source in hosts:
        for dest in hosts:
            if source != dest:
                total_tests += 1
                ping_result = source.cmd('ping -c 2 -W 1 {0}'.format(dest.IP()))
                if '0% packet loss' in ping_result:
                    print("PASS: {0} -> {1} ({2}): SUCCESS\n".format(source.name, dest.name, dest.IP()))
                    successful_tests += 1
                else:
                    print("FAIL: {0} -> {1} ({2}): FAILED\n".format(source.name, dest.name, dest.IP()))
    
    success_rate = (successful_tests / total_tests) * 100 if total_tests > 0 else 0
    print("*** Connectivity tests completed: {0}/{1} successful ({2:.1f}%)\n".format(successful_tests, total_tests, success_rate))

def test_bandwidth(net):
    print("*** Testing bandwidth between selected hosts\n")

    h1, h5 = net.get('h1', 'h5')  
    
    print("*** Starting iperf server on {0}\n".format(h5.name))
    h5.cmd('iperf -s &')  
    time.sleep(1)  
    
    print("*** Running bandwidth test from {0} to {1}\n".format(h1.name, h5.name))
    iperf_result = h1.cmd('iperf -c {0} -t 5'.format(h5.IP()))
    print("*** Bandwidth test results:\n{0}\n".format(iperf_result))
    
    h5.cmd('kill %iperf')
    
    bandwidth = "Unknown"
    for line in iperf_result.split('\n'):
        if 'Mbits/sec' in line and 'sender' in line:
            bandwidth = line.split('Mbits/sec')[0].split()[-1] + " Mbits/sec"
    
    print("*** Bandwidth between {0} and {1}: {2}\n".format(h1.name, h5.name, bandwidth))

def test_latency(net):
    print("*** Testing latency between all routers\n")
    routers = [h for h in net.hosts if h.name.startswith('r')]
    
    for source in routers:
        for dest in routers:
            if source != dest:
                print("*** Measuring latency from {0} to {1}\n".format(source.name, dest.name))
                ping_result = source.cmd('ping -c 5 {0}'.format(dest.IP()))
                
                avg_latency = "Unknown"
                for line in ping_result.split('\n'):
                    if 'min/avg/max' in line:
                        avg_latency = line.split('=')[1].split('/')[1].strip() + " ms"
                
                print("*** Latency {0} -> {1}: {2}\n".format(source.name, dest.name, avg_latency))

def check_routing_tables(net):
    print("*** Checking routing tables on all routers\n")
    routers = [h for h in net.hosts if h.name.startswith('r')]
    
    for router in routers:
        all_correct = True
        print("*** Checking routing table for {0}:\n".format(router.name))
        routes = router.cmd('ip route')
        
        for i in range(1, 6):  
            subnet = "10.0.{0}.0/24".format(i)
            if subnet not in routes and not router.name == 'r{0}'.format(i):
                print("FAIL: Missing route to {0} in {1}\n".format(subnet, router.name))
                all_correct = False
        if not all_correct:
            print("PASS: All required routes are present in routing table")

def test_path_tracing(net):
    print("*** Tracing paths between hosts in different subnets\n")
    
    h1, h7 = net.get('h1', 'h7') 
    
    print("*** Tracing path from {0} to {1}\n".format(h1.name, h7.name))
    traceroute = h1.cmd('traceroute -n {0}'.format(h7.IP()))
    print("{0}\n".format(traceroute))

def run_all_tests():
    try:
        setLogLevel('info')
        net = topology()
        
        print('*** Waiting for network to stabilize\n')
        time.sleep(2)
        
        # Dump connections for debugging
        print('\n*** Dumping host connections\n')
        dumpNodeConnections(net.hosts)
        
        # Run tests
        print('\n*** STARTING NETWORK TESTS ***\n\n')
        
        test_connectivity(net)
        test_bandwidth(net)
        test_latency(net)
        check_routing_tables(net)
        test_path_tracing(net)
        
    except Exception as e:
        print("*** Error during test execution: {0}\n".format(e))
        traceback.print_exc()
    finally:
        if net:
            print('*** Stopping network\n')
            net.stop()

if __name__ == '__main__':
    run_all_tests()