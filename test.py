from mininet.log import setLogLevel, info
from mininet.util import dumpNodeConnections
import time
from topology import topology
import traceback
from threading import Thread

def test_connectivity(net):
    print("\n*** Testing connectivity between all hosts\n")
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
                    print("PASS: {0} -> {1}: SUCCESS".format(source.name, dest.name))
                    successful_tests += 1
                else:
                    print("FAIL: {0} -> {1}: FAILED".format(source.name, dest.name))
    
    success_rate = (successful_tests / total_tests) * 100 if total_tests > 0 else 0
    print("*** Connectivity tests completed: {0}/{1} successful ({2:.1f}%)".format(successful_tests, total_tests, success_rate))

def test_bandwidth(net):
    print("\n*** Testing bandwidth between selected hosts\n")

    h1, h5 = net.get('h1', 'h5')  
    
    print("*** Starting iperf server on {0}".format(h5.name))
    h5.cmd('iperf -s &')  
    time.sleep(1)  
    
    print("*** Running bandwidth test from {0} to {1}".format(h1.name, h5.name))
    iperf_result = h1.cmd('iperf -c {0} -t 5'.format(h5.IP()))
    print("*** Bandwidth test results:\n{0}\n".format(iperf_result))
    
    h5.cmd('kill %iperf')
    
    bandwidth = "Unknown"
    for line in iperf_result.split('\n'):
        if 'Mbits/sec' in line:
            bandwidth = line.split('Mbits/sec')[0].split()[-1] + " Mbits/sec"
            break
        elif 'Gbits/sec' in line:
            bandwidth = line.split('Gbits/sec')[0].split()[-1] + " Gbits/sec"
    
    print("*** Bandwidth between {0} and {1}: {2}".format(h1.name, h5.name, bandwidth))

def test_latency(net):
    print("\n*** Testing latency between all routers\n")
    routers = [h for h in net.hosts if h.name.startswith('r')]
    
    for source in routers:
        for dest in routers:
            if source != dest:
                ping_result = source.cmd('ping -c 5 {0}'.format(dest.IP()))
                
                avg_latency = "Unknown"
                for line in ping_result.split('\n'):
                    if 'min/avg/max' in line:
                        avg_latency = line.split('=')[1].split('/')[1].strip() + " ms"
                
                print("*** Latency {0} -> {1}: {2}".format(source.name, dest.name, avg_latency))

def check_routing_tables(net):
    print("\n*** Checking routing tables on all routers\n")
    routers = [h for h in net.hosts if h.name.startswith('r')]
    
    for router in routers:
        all_correct = True
        print("*** Checking routing table for {0}:".format(router.name))
        routes = router.cmd('ip route')
        
        for i in range(1, 6):  
            subnet = "10.0.{0}.0/24".format(i)
            if subnet not in routes and not router.name == 'r{0}'.format(i):
                print(" FAIL: Missing route to {0} in {1}".format(subnet, router.name))
                all_correct = False
        if all_correct:
            print(" PASS: All required routes are present in routing table\n")

def test_path_tracing(net):
    print("*** Tracing paths between hosts in different subnets\n")
    
    h1, h7 = net.get('h1', 'h7') 
    
    print("*** Tracing path from {0} to {1}\n".format(h1.name, h7.name))
    traceroute = h1.cmd('traceroute -n {0}'.format(h7.IP()))
    print("{0}".format(traceroute))

def test_simultaneous_traffic(net, num_pairs=3, duration=10):
    """
    Test simultaneous network traffic between multiple pairs of hosts.
    
    Args:
        net: Mininet network object
        num_pairs: Number of host pairs to test simultaneously (default: 3)
        duration: Duration of the test in seconds (default: 10)
    """
    import random
    from threading import Thread
    
    print("\n*** Testing simultaneous network traffic between {0} host pairs for {1} seconds\n".format(num_pairs, duration))
    
    # Get all hosts and filter out routers
    all_hosts = [h for h in net.hosts if h.name.startswith('h')]
    
    if len(all_hosts) < num_pairs * 2:
        print("WARNING: Not enough hosts for {0} pairs. Using {1} pairs instead.".format(
            num_pairs, len(all_hosts) // 2))
        num_pairs = len(all_hosts) // 2
    
    # Select random pairs of hosts
    host_pairs = []
    hosts_copy = all_hosts.copy()
    random.shuffle(hosts_copy)
    
    for i in range(num_pairs):
        if len(hosts_copy) >= 2:
            source = hosts_copy.pop()
            dest = hosts_copy.pop()
            host_pairs.append((source, dest))
    
    # Start iperf servers on destination hosts
    for _, dest in host_pairs:
        dest.cmd('iperf -s &')
    
    print("*** Started {0} iperf servers".format(len(host_pairs)))
    time.sleep(1)  # Give servers time to start
    
    # Create threads to run iperf clients
    threads = []
    results = {}
    
    def run_iperf_client(source, dest, pair_id):
        print("*** Starting traffic from {0} to {1}".format(source.name, dest.name))
        iperf_result = source.cmd('iperf -c {0} -t {1} -i 1'.format(dest.IP(), duration))
        results[pair_id] = iperf_result
    
    # Start all client threads simultaneously
    for i, (source, dest) in enumerate(host_pairs):
        thread = Thread(target=run_iperf_client, args=(source, dest, i))
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    print("\n*** All traffic tests completed")
    
    # Process and display results
    print("\n*** SIMULTANEOUS TRAFFIC TEST RESULTS ***")
    
    total_bandwidth = 0
    for i, (source, dest) in enumerate(host_pairs):
        iperf_result = results[i]
        
        # Extract bandwidth from results
        bandwidth = "Unknown"
        for line in iperf_result.split('\n'):
            if 'Mbits/sec' in line and 'sec' in line and '-' in line:
                # Look for the summary line
                if '0.0-' in line:
                    bandwidth = line.split('Mbits/sec')[0].split()[-1]
                    try:
                        total_bandwidth += float(bandwidth)
                    except ValueError:
                        pass
                    bandwidth += " Mbits/sec"
                    break
        
        # Measure latency under load
        ping_result = source.cmd('ping -c 3 -i 0.2 {0}'.format(dest.IP()))
        
        avg_latency = "Unknown"
        packet_loss = "Unknown"
        
        for line in ping_result.split('\n'):
            if 'min/avg/max' in line:
                avg_latency = line.split('=')[1].split('/')[1].strip() + " ms"
            if 'packet loss' in line:
                packet_loss = line.split(',')[2].strip()
        
        print("Pair {0}: {1} -> {2}".format(i+1, source.name, dest.name))
        print("  - Bandwidth: {0}".format(bandwidth))
        print("  - Latency under load: {0}".format(avg_latency))
        print("  - Packet loss: {0}".format(packet_loss))
    
    # Kill all iperf servers
    for _, dest in host_pairs:
        dest.cmd('kill %iperf')
    
    print("\n*** Aggregate network bandwidth: {0:.2f} Mbits/sec".format(total_bandwidth))
    
    # Test network congestion by pinging between hosts not involved in iperf
    remaining_hosts = hosts_copy
    if len(remaining_hosts) >= 2:
        print("\n*** Testing network congestion on uninvolved hosts")
        h_source = remaining_hosts[0]
        h_dest = remaining_hosts[1]
        
        print("Measuring latency between {0} and {1} during traffic tests".format(
            h_source.name, h_dest.name))
        
        ping_result = h_source.cmd('ping -c 5 {0}'.format(h_dest.IP()))
        
        avg_latency = "Unknown"
        for line in ping_result.split('\n'):
            if 'min/avg/max' in line:
                avg_latency = line.split('=')[1].split('/')[1].strip() + " ms"
        
        print("Latency on uncongested path: {0}".format(avg_latency))

def run_all_tests():
    try:
        setLogLevel('info')
        net = topology()
        
        print('*** Waiting for network to stabilize\n')
        time.sleep(2)
        
        # Dump connections for debugging
        print('*** Dumping host connections\n')
        dumpNodeConnections(net.hosts)
        
        # Run tests
        print('\n*** STARTING NETWORK TESTS ***')
        
        test_connectivity(net)
        test_bandwidth(net)
        test_latency(net)
        check_routing_tables(net)
        test_path_tracing(net)
        test_simultaneous_traffic(net, num_pairs=3, duration=10)
        
    except Exception as e:
        print("*** Error during test execution: {0}\n".format(e))
        traceback.print_exc()
    finally:
        if net:
            print('*** Stopping network\n')
            net.stop()

if __name__ == '__main__':
    run_all_tests()