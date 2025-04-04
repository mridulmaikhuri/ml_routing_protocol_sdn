from mininet.net import Mininet
from mininet.node import Controller
from mininet.node import Host, Node
from mininet.node import OVSKernelSwitch
from mininet.log import setLogLevel, info
from mininet.util import dumpNodeConnections
import os
import time
import sys

def test_connectivity(net):
    """
    Test connectivity between all hosts in the network
    """
    info("*** Testing basic connectivity between all hosts\n")
    hosts = net.hosts
    total_tests = 0
    successful_tests = 0
    
    for source in hosts:
        for dest in hosts:
            if source != dest:
                total_tests += 1
                ping_result = source.cmd('ping -c 3 -W 1 {0}'.format(dest.IP()))
                if '0% packet loss' in ping_result:
                    info("PASS: {0} -> {1} ({2}): SUCCESS\n".format(source.name, dest.name, dest.IP()))
                    successful_tests += 1
                else:
                    info("FAIL: {0} -> {1} ({2}): FAILED\n".format(source.name, dest.name, dest.IP()))
    
    success_rate = (successful_tests / total_tests) * 100 if total_tests > 0 else 0
    info("*** Connectivity tests completed: {0}/{1} successful ({2:.1f}%)\n".format(successful_tests, total_tests, success_rate))
    return successful_tests == total_tests

def test_bandwidth(net):
    """
    Test bandwidth between selected hosts using iperf
    """
    info("*** Testing bandwidth between selected hosts\n")
    
    # Select some hosts for bandwidth testing
    # Test between hosts in different subnets
    h1, h5 = net.get('h1', 'h5')  # h1 is in subnet 1, h5 is in subnet 3
    
    info("*** Starting iperf server on {0}\n".format(h5.name))
    h5.cmd('iperf -s &')  # Start iperf server in background
    time.sleep(1)  # Wait for server to start
    
    info("*** Running bandwidth test from {0} to {1}\n".format(h1.name, h5.name))
    iperf_result = h1.cmd('iperf -c {0} -t 5'.format(h5.IP()))
    info("*** Bandwidth test results:\n{0}\n".format(iperf_result))
    
    # Kill iperf server
    h5.cmd('kill %iperf')
    
    # Extract bandwidth from results
    bandwidth = "Unknown"
    for line in iperf_result.split('\n'):
        if 'Mbits/sec' in line and 'sender' in line:
            bandwidth = line.split('Mbits/sec')[0].split()[-1] + " Mbits/sec"
    
    info("*** Bandwidth between {0} and {1}: {2}\n".format(h1.name, h5.name, bandwidth))
    return True  # Return test result

def test_latency(net):
    """
    Test latency between all routers in the network
    """
    info("*** Testing latency between all routers\n")
    routers = [h for h in net.hosts if h.name.startswith('r')]
    
    for source in routers:
        for dest in routers:
            if source != dest:
                info("*** Measuring latency from {0} to {1}\n".format(source.name, dest.name))
                ping_result = source.cmd('ping -c 5 {0}'.format(dest.IP()))
                
                # Extract average latency
                avg_latency = "Unknown"
                for line in ping_result.split('\n'):
                    if 'min/avg/max' in line:
                        avg_latency = line.split('=')[1].split('/')[1].strip() + " ms"
                
                info("*** Latency {0} -> {1}: {2}\n".format(source.name, dest.name, avg_latency))
    
    return True

def test_routing_tables(net):
    """
    Test that routing tables are correctly configured on all routers
    """
    info("*** Checking routing tables on all routers\n")
    routers = [h for h in net.hosts if h.name.startswith('r')]
    
    all_correct = True
    for router in routers:
        info("*** Routing table for {0}:\n".format(router.name))
        routes = router.cmd('ip route')
        info("{0}\n".format(routes))
        
        # Check if each subnet is reachable
        for i in range(1, 6):  # Check routes to all 5 subnets
            subnet = "10.0.{0}.0/24".format(i)
            if subnet not in routes and not router.name == 'r{0}'.format(i):
                info("FAIL: Missing route to {0} in {1}\n".format(subnet, router.name))
                all_correct = False
    
    if all_correct:
        info("PASS: All required routes present in routing tables\n")
    else:
        info("FAIL: Some required routes are missing\n")
    
    return all_correct

def test_path_tracing(net):
    """
    Trace paths between different hosts to verify routing
    """
    info("*** Tracing paths between hosts in different subnets\n")
    
    # Select hosts from different subnets
    h1, h7 = net.get('h1', 'h7')  # h1 in subnet 1, h7 in subnet 4
    
    info("*** Tracing path from {0} to {1}\n".format(h1.name, h7.name))
    traceroute = h1.cmd('traceroute -n {0}'.format(h7.IP()))
    info("{0}\n".format(traceroute))
    
    # Check if the path goes through the expected routers
    r1_ip = "10.0.1.1"
    if r1_ip not in traceroute:
        info("FAIL: Path does not go through {0} (r1) as expected\n".format(r1_ip))
        return False
    
    info("PASS: Path verification successful\n")
    return True

def run_all_tests():
    """
    Run a custom network test script for the mesh topology
    """
    # Create the network using the topology function
    net = None
    
    try:
        # Import and create the topology
        setLogLevel('info')
        
        # Create our network
        info('*** Creating the network topology\n')
        
        # Initialize Mininet
        net = Mininet(topo=None, build=False, ipBase='10.0.0.0/8')
        
        # Add controller
        info('*** Adding controller\n')
        c0 = net.addController(name='c0', controller=Controller, protocol='tcp', port=6633)
        
        # Define number of routers and switches
        num = 5
        
        # Add routers
        routers = []
        info('*** Adding routers\n')
        for i in range(num):
            router_name = 'r{0}'.format(i+1)
            router = net.addHost(router_name, cls=Node, ip='10.0.{0}.1/24'.format(i+1))
            router.cmd('sysctl -w net.ipv4.ip_forward=1')       
            routers.append(router)
        
        # Add switches
        switches = []
        info('*** Adding switches\n')
        for i in range(num):
            switch_name = 's{0}'.format(i+1)
            switch = net.addSwitch(switch_name, cls=OVSKernelSwitch)
            switches.append(switch)
        
        # Add hosts 
        info('*** Adding hosts\n')
        hosts = []
        for i in range(num):
            for j in range(2):
                host_index = i*2 + j + 1
                host_name = 'h{0}'.format(host_index)
                host = net.addHost(host_name, cls=Host, 
                                 ip='10.0.{0}.{1}/24'.format(i+1, 100+j), 
                                 defaultRoute='via 10.0.{0}.1'.format(i+1))
                hosts.append(host)
                net.addLink(host, switches[i])
        
        # Connect switches to their respective routers
        info('*** Creating links between switches and routers\n')
        for i in range(num):
            net.addLink(switches[i], routers[i], 
                       intfName2='r{0}-eth0'.format(i+1), 
                       params2={'ip': '10.0.{0}.1/24'.format(i+1)})
        
        # Create mesh topology by connecting all routers to each other
        info('*** Creating mesh links between routers\n')
        link_count = 0
        for i in range(num):
            for j in range(i+1, num):
                link_count += 1
                subnet = 10 + link_count 
                
                # Add link between routers
                intfName1 = 'r{0}-eth{1}'.format(i+1, j+1)
                intfName2 = 'r{0}-eth{1}'.format(j+1, i+1)
                
                net.addLink(
                    routers[i], 
                    routers[j],
                    intfName1=intfName1, 
                    intfName2=intfName2,
                    params1={'ip': '10.0.{0}.{1}/24'.format(subnet, i+1)},
                    params2={'ip': '10.0.{0}.{1}/24'.format(subnet, j+1)}
                )
                
                # Add routes
                routers[i].cmd('ip route add 10.0.{0}.0/24 via 10.0.{1}.{2}'.format(j+1, subnet, j+1))
                routers[j].cmd('ip route add 10.0.{0}.0/24 via 10.0.{1}.{2}'.format(i+1, subnet, i+1))
        
        # Start network
        info('*** Starting network\n')
        net.build()
        
        # Start controller
        c0.start()
        
        # Start switches
        for switch in switches:
            switch.start([c0])
        
        # Give the network time to start
        info('*** Waiting for network to stabilize\n')
        time.sleep(2)
        
        # Dump connections for debugging
        info('*** Dumping host connections\n')
        dumpNodeConnections(net.hosts)
        
        # Run tests
        info('\n*** STARTING NETWORK TESTS ***\n\n')
        
        # Test 1: Basic connectivity
        test_results = []
        test_results.append(("Connectivity Test", test_connectivity(net)))
        
        # Test 2: Bandwidth
        test_results.append(("Bandwidth Test", test_bandwidth(net)))
        
        # Test 3: Latency
        test_results.append(("Latency Test", test_latency(net)))
        
        # Test 4: Routing Tables
        test_results.append(("Routing Tables Test", test_routing_tables(net)))
        
        # Test 5: Path Tracing
        test_results.append(("Path Tracing Test", test_path_tracing(net)))
        
        # Print test summary
        info('\n\n*** TEST SUMMARY ***\n')
        all_passed = True
        for test_name, result in test_results:
            status = "PASSED" if result else "FAILED"
            info("{0}: {1}\n".format(test_name, status))
            all_passed = all_passed and result
        
        overall_status = "PASSED" if all_passed else "FAILED"
        info("\nOverall Test Result: {0}\n".format(overall_status))
        
    except Exception as e:
        info("*** Error during test execution: {0}\n".format(e))
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        if net:
            info('*** Stopping network\n')
            net.stop()

if __name__ == '__main__':
    run_all_tests()