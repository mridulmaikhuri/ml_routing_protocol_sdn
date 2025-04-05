from mininet.log import setLogLevel, info
from mininet.util import dumpNodeConnections
import time
from topology import topology
import traceback
import random
from threading import Thread

def test_connectivity(net):
    print("\n*** Testing connectivity between all hosts \n")
    host_names = ['h{0}'.format(i + 1) for i in range(10)]
    hosts = [net.get(h) for h in host_names]
    total_tests = 0
    successful_tests = 0
    total_latency = 0.0  # Changed to float
    total_bandwidth = 0.0  # Changed to float
    
    for i in range(len(hosts)):
        for j in range(i + 1, len(hosts)):
            source = hosts[i]
            dest = hosts[j]
            print('{0} -> {1}'.format(source.name, dest.name))
            total_tests += 1
            
            # Ping test
            ping_result = source.cmd('ping -c 2 -W 1 {0}'.format(dest.IP()))
            print(' packet loss: {0}'.format(ping_result.split(' packet loss')[0].split()[-1]))

            # Latency calculation (extract numeric value)
            avg_latency = 0.0
            for line in ping_result.split('\n'):
                if 'min/avg/max' in line:
                    avg_latency = float(line.split('=')[1].split('/')[1].strip())
                    break
            
            total_latency += avg_latency
            print(" Latency: {0} ms".format(avg_latency))

            # Bandwidth test
            dest.cmd('iperf -s &')
            time.sleep(1)
            iperf_result = source.cmd('iperf -c {0} -t 5'.format(dest.IP()))
            dest.cmd('kill %iperf')
            
            # Bandwidth calculation (extract numeric value)
            bandwidth = 0.0
            for line in iperf_result.split('\n'):
                if 'Mbits/sec' in line:
                    bandwidth = float(line.split('Mbits/sec')[0].split()[-1])
                    break
                elif 'Gbits/sec' in line:
                    bandwidth = float(line.split('Gbits/sec')[0].split()[-1]) * 1000  # Convert Gbps to Mbps
                    break
            
            total_bandwidth += bandwidth
            print(' bandwidth: {0} Mbits/sec'.format(bandwidth))
            
            if avg_latency > 0:  # Consider test successful if we got latency data
                successful_tests += 1
    
    # Calculate averages
    success_rate = (successful_tests / total_tests) * 100 if total_tests > 0 else 0
    avg_latency = total_latency / successful_tests if successful_tests > 0 else 0
    avg_bandwidth = total_bandwidth / successful_tests if successful_tests > 0 else 0

    print("\n*** Connectivity tests completed")
    print(" Success Rate: {0:.1f}%".format(success_rate))
    print(" Average Latency: {0:.2f} ms".format(avg_latency))
    print(" Average Bandwidth: {0:.2f} Mbits/sec".format(avg_bandwidth))

def test_simultaneous_traffic(net, num_pairs=3, duration=10):  
    print("\n*** Testing simultaneous network traffic between {0} host pairs for {1} seconds\n".format(num_pairs, duration))
    
    all_hosts = [h for h in net.hosts if h.name.startswith('h')]
    
    host_pairs = []
    hosts_copy = all_hosts[:] 
    random.shuffle(hosts_copy)
    
    for i in range(num_pairs):
        if len(hosts_copy) >= 2:
            source = hosts_copy.pop()
            dest = hosts_copy.pop()
            host_pairs.append((source, dest))
    
    for _, dest in host_pairs:
        dest.cmd('iperf -s &')
    
    print("*** Started {0} iperf servers".format(len(host_pairs)))
    time.sleep(1) 
    
    threads = []
    results = {}
    
    def run_iperf_client(source, dest, pair_id):
        print("*** Starting traffic from {0} to {1}".format(source.name, dest.name))
        iperf_result = source.cmd('iperf -c {0} -t {1} -i 1'.format(dest.IP(), duration))
        results[pair_id] = iperf_result
    
    for i, (source, dest) in enumerate(host_pairs):
        thread = Thread(target=run_iperf_client, args=(source, dest, i))
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()
    
    print("\n*** All traffic tests completed")
    
    print("\n*** SIMULTANEOUS TRAFFIC TEST RESULTS ***")
    
    total_bandwidth = 0
    total_latency = 0
    total_loss = 0
    for i, (source, dest) in enumerate(host_pairs):
        iperf_result = results[i]
        
        bandwidth = 0.0
        for line in iperf_result.split('\n'):
            if 'Mbits/sec' in line:
                bandwidth = float(line.split('Mbits/sec')[0].split()[-1])
                break
            elif 'Gbits/sec' in line:
                bandwidth = float(line.split('Gbits/sec')[0].split()[-1]) * 1000  
                break
        
        total_bandwidth += bandwidth
        
        ping_result = source.cmd('ping -c 3 -i 0.2 {0}'.format(dest.IP()))
        
        avg_latency = "Unknown"
        packet_loss = "Unknown"
        
        for line in ping_result.split('\n'):
            if 'min/avg/max' in line:
                avg_latency = float(line.split('=')[1].split('/')[1].strip())
                total_latency += avg_latency
            if 'packet loss' in line:
                packet_loss = line.split(',')[2].strip()
                total_loss += float(packet_loss.split('%')[0])
        
        print("Pair {0}: {1} -> {2}".format(i+1, source.name, dest.name))
        print("  - Bandwidth: {0}".format(bandwidth))
        print("  - Latency under load: {0} ms".format(avg_latency))
        print("  - Packet loss: {0}".format(packet_loss))

    for _, dest in host_pairs:
        dest.cmd('kill %iperf')
    
    print("\n*** test results")
    print(" average packet loss: {0}".format(total_loss/num_pairs))
    print(" average latency: {0}".format(total_latency/num_pairs))
    print(" average bandwidth: {0}".format(total_bandwidth/num_pairs))

def test_fault_tolerance(net, num_failures=1):
    print("\n*** Testing network fault tolerance with {0} link failure(s) ***\n".format(num_failures))
    
    # Store original network state for baseline measurements
    all_hosts = [h for h in net.hosts if h.name.startswith('h')]
    all_links = net.links
    
    # Select random host pairs for testing - one pair from different subnets
    host_pairs = []
    
    # Try to select hosts from different subnets if possible
    subnet1_hosts = [h for h in all_hosts if h.IP().startswith('10.0.1')]
    subnet2_hosts = [h for h in all_hosts if h.IP().startswith('10.0.2')]
    
    if subnet1_hosts and subnet2_hosts:
        h1 = random.choice(subnet1_hosts)
        h2 = random.choice(subnet2_hosts)
        host_pairs.append((h1, h2))
    else:
        # Fallback if subnet identification fails
        if len(all_hosts) >= 2:
            hosts_sample = random.sample(all_hosts, 2)
            host_pairs.append((hosts_sample[0], hosts_sample[1]))
    
    # Select two more random pairs
    remaining_hosts = [h for h in all_hosts if not any(h in pair for pair in host_pairs)]
    for _ in range(min(2, len(remaining_hosts) // 2)):
        if len(remaining_hosts) >= 2:
            h1 = remaining_hosts.pop(random.randrange(len(remaining_hosts)))
            h2 = remaining_hosts.pop(random.randrange(len(remaining_hosts)))
            host_pairs.append((h1, h2))
    
    # Run baseline tests
    print("*** Running baseline performance tests")
    baseline_results = {}
    
    for i, (source, dest) in enumerate(host_pairs):
        print("Pair {0}: Testing baseline performance from {1} to {2}".format(
            i+1, source.name, dest.name))
        
        # Test connectivity
        ping_result = source.cmd('ping -c 3 -W 1 {0}'.format(dest.IP()))
        connectivity = '0% packet loss' in ping_result
        
        # Extract latency
        latency = "N/A"
        for line in ping_result.split('\n'):
            if 'min/avg/max' in line:
                latency = line.split('=')[1].split('/')[1].strip() + " ms"
        
        # Test bandwidth
        dest.cmd('iperf -s &')
        time.sleep(1)
        bandwidth_result = source.cmd('iperf -c {0} -t 3'.format(dest.IP()))
        dest.cmd('kill %iperf')
        
        bandwidth = "N/A"
        for line in bandwidth_result.split('\n'):
            if 'Mbits/sec' in line and '0.0-' in line:
                bandwidth = line.split('Mbits/sec')[0].split()[-1] + " Mbits/sec"
        
        # Get traceroute path
        traceroute = source.cmd('traceroute -n -w 1 -q 1 {0}'.format(dest.IP()))
        
        baseline_results[i] = {
            'connectivity': connectivity,
            'latency': latency,
            'bandwidth': bandwidth,
            'traceroute': traceroute
        }
    
    # Function to fail links and test impact
    def test_with_failed_links(links_to_fail):
        # Disable the selected links
        for link in links_to_fail:
            src, dst = link.intf1.node, link.intf2.node
            print("*** Failing link between {0} and {1}".format(
                src.name, dst.name))
            
            # Bring down both interfaces
            src.cmd('ifconfig {0} down'.format(link.intf1.name))
            dst.cmd('ifconfig {0} down'.format(link.intf2.name))
        
        # Wait for routing protocols to converge
        print("*** Waiting for network to reconverge")
        time.sleep(5)
        
        # Run tests with failed links
        failure_results = {}
        
        for i, (source, dest) in enumerate(host_pairs):
            print("Pair {0}: Testing performance from {1} to {2} with failed links".format(
                i+1, source.name, dest.name))
            
            # Test connectivity
            ping_result = source.cmd('ping -c 3 -W 1 {0}'.format(dest.IP()))
            connectivity = '0% packet loss' in ping_result
            
            # Extract latency
            latency = "N/A"
            for line in ping_result.split('\n'):
                if 'min/avg/max' in line:
                    latency = line.split('=')[1].split('/')[1].strip() + " ms"
            
            # Test bandwidth if still connected
            bandwidth = "N/A"
            if connectivity:
                dest.cmd('iperf -s &')
                time.sleep(1)
                bandwidth_result = source.cmd('iperf -c {0} -t 3'.format(dest.IP()))
                dest.cmd('kill %iperf')
                
                for line in bandwidth_result.split('\n'):
                    if 'Mbits/sec' in line and '0.0-' in line:
                        bandwidth = line.split('Mbits/sec')[0].split()[-1] + " Mbits/sec"
            
            # Get new traceroute path if still connected
            traceroute = "N/A"
            if connectivity:
                traceroute = source.cmd('traceroute -n -w 1 -q 1 {0}'.format(dest.IP()))
            
            failure_results[i] = {
                'connectivity': connectivity,
                'latency': latency,
                'bandwidth': bandwidth,
                'traceroute': traceroute
            }
        
        return failure_results
    
    # Get links between routers (or other links if no router links exist)
    router_links = [link for link in all_links 
                   if (hasattr(link.intf1.node, 'name') and 
                       hasattr(link.intf2.node, 'name') and
                       link.intf1.node.name.startswith('r') and 
                       link.intf2.node.name.startswith('r'))]
    
    # If no router links, fall back to any links
    links_to_test = router_links if router_links else all_links
    
    # Ensure we don't try to fail more links than exist
    num_failures = min(num_failures, len(links_to_test))
    
    # Select random links to fail
    failed_links = random.sample(links_to_test, num_failures)
    
    # Test with failed links
    failure_results = test_with_failed_links(failed_links)
    
    # Restore network
    print("\n*** Restoring network links")
    for link in failed_links:
        src, dst = link.intf1.node, link.intf2.node
        src.cmd('ifconfig {0} up'.format(link.intf1.name))
        dst.cmd('ifconfig {0} up'.format(link.intf2.name))
    
    # Wait for network to recover
    print("*** Waiting for network to recover")
    time.sleep(5)
    
    # Print comparison results
    print("\n*** FAULT TOLERANCE TEST RESULTS ***")
    print("Failed links: {0}".format(", ".join([f"{link.intf1.node.name}-{link.intf2.node.name}" 
                                             for link in failed_links])))
    
    for i, (source, dest) in enumerate(host_pairs):
        print("\nPair {0}: {1} -> {2}".format(i+1, source.name, dest.name))
        
        baseline = baseline_results[i]
        failure = failure_results[i]
        
        print("  Connectivity before failure: {0}".format("YES" if baseline['connectivity'] else "NO"))
        print("  Connectivity after failure: {0}".format("YES" if failure['connectivity'] else "NO"))
        
        if baseline['connectivity'] and failure['connectivity']:
            # Calculate latency change if both are available
            try:
                baseline_latency = float(baseline['latency'].split()[0])
                failure_latency = float(failure['latency'].split()[0])
                latency_change = ((failure_latency - baseline_latency) / baseline_latency) * 100
                print("  Latency before: {0}".format(baseline['latency']))
                print("  Latency after: {0} ({1:+.1f}%)".format(
                    failure['latency'], latency_change))
            except (ValueError, IndexError):
                print("  Latency before: {0}".format(baseline['latency']))
                print("  Latency after: {0}".format(failure['latency']))
            
            # Calculate bandwidth change if both are available
            try:
                baseline_bw = float(baseline['bandwidth'].split()[0])
                failure_bw = float(failure['bandwidth'].split()[0])
                bw_change = ((failure_bw - baseline_bw) / baseline_bw) * 100
                print("  Bandwidth before: {0}".format(baseline['bandwidth']))
                print("  Bandwidth after: {0} ({1:+.1f}%)".format(
                    failure['bandwidth'], bw_change))
            except (ValueError, IndexError):
                print("  Bandwidth before: {0}".format(baseline['bandwidth']))
                print("  Bandwidth after: {0}".format(failure['bandwidth']))
            
            # Compare traceroutes if both are available
            if baseline['traceroute'] != "N/A" and failure['traceroute'] != "N/A":
                print("  Path before failure:")
                for line in baseline['traceroute'].split('\n'):
                    if line.strip() and not line.startswith('traceroute'):
                        print(f"    {line}")
                
                print("  Path after failure:")
                for line in failure['traceroute'].split('\n'):
                    if line.strip() and not line.startswith('traceroute'):
                        print(f"    {line}")
        
        # Overall resilience assessment
        if not baseline['connectivity']:
            print("  ASSESSMENT: Pair was disconnected before link failure")
        elif not failure['connectivity']:
            print("  ASSESSMENT: FAILED - Connection lost after link failure")
        else:
            try:
                baseline_latency = float(baseline['latency'].split()[0])
                failure_latency = float(failure['latency'].split()[0])
                latency_change = ((failure_latency - baseline_latency) / baseline_latency) * 100
                
                if latency_change > 100:  # More than doubled
                    print("  ASSESSMENT: DEGRADED - Significant performance impact")
                elif latency_change > 20:  # More than 20% increase
                    print("  ASSESSMENT: AFFECTED - Noticeable performance impact")
                else:
                    print("  ASSESSMENT: RESILIENT - Minimal performance impact")
            except (ValueError, IndexError):
                print("  ASSESSMENT: UNKNOWN - Could not calculate performance impact")
    

def run_all_tests():
    net = None
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
        test_simultaneous_traffic(net, num_pairs=3, duration=10)
        test_fault_tolerance(net, num_failures=2)
        
    except Exception as e:
        print("*** Error during test execution: {0}\n".format(e))
        traceback.print_exc()
    finally:
        if net:
            print('\n*** Stopping network\n')
            net.stop()

if __name__ == '__main__':
    run_all_tests()