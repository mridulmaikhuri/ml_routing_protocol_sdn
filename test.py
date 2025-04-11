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
    total_latency = 0.0  
    total_bandwidth = 0.0  
    
    for i in range(len(hosts)):
        for j in range(i + 1, len(hosts)):
            source = hosts[i]
            dest = hosts[j]
            print('{0} -> {1}'.format(source.name, dest.name))
            total_tests += 1
            
            # Ping test
            ping_result = source.cmd('ping -c 2 -W 1 {0}'.format(dest.IP()))
            print(' packet loss: {0}'.format(ping_result.split(' packet loss')[0].split()[-1]))

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
            
            bandwidth = 0.0
            for line in iperf_result.split('\n'):
                if 'Mbits/sec' in line:
                    bandwidth = float(line.split('Mbits/sec')[0].split()[-1])
                    break
                elif 'Gbits/sec' in line:
                    bandwidth = float(line.split('Gbits/sec')[0].split()[-1]) * 1000 
                    break
            
            total_bandwidth += bandwidth
            print(' bandwidth: {0} Mbits/sec'.format(bandwidth))
            
            if avg_latency > 0:  
                successful_tests += 1
    
    # Calculate averages
    success_rate = (successful_tests / total_tests) * 100 if total_tests > 0 else 0
    avg_latency = total_latency / successful_tests if successful_tests > 0 else 0
    avg_bandwidth = total_bandwidth / successful_tests if successful_tests > 0 else 0

    print("\n*** Connectivity tests completed")
    print(" Success Rate: {0:.2f}%".format(success_rate))
    print(" Average Latency: {0:.2f} ms".format(avg_latency))
    print(" Average Bandwidth: {0:.2f} Mbits/sec".format(avg_bandwidth))

def test_simultaneous_traffic(net, duration=10):  
    print("\n*** Testing simultaneous network traffic\n")
    
    host_pairs = []
    for i in range(0, 10, 2):
        host_pairs.append((net.get('h{0}'.format(i + 1)), net.get('h{0}'.format(10 - i))))
    
    for _, dest in host_pairs:
        dest.cmd('iperf -s &')
    
    print("*** Started {0} iperf servers".format(len(host_pairs)))
    time.sleep(1) 
    
    threads = []
    results = {}
    
    def run_iperf_client(source, dest, pair_id):
        print("*** Starting traffic from {0} to {1}".format(source.name, dest.name))
        
        # ping test
        ping_cmd = 'ping -c {0} -i 0.2 {1} > /tmp/ping_results_{2} 2>&1 &'.format(
            duration * 5,  
            dest.IP(), 
            pair_id
        )
        source.cmd(ping_cmd)
        ping_pid = source.cmd("echo $!")
        
        # Bandwidth test
        iperf_result = source.cmd('iperf -c {0} -t {1} -i 1'.format(dest.IP(), duration))
        
        bandwidth = 0.0
        for line in iperf_result.split('\n'):
            if 'Mbits/sec' in line and 'sec' in line.split('Mbits/sec')[0]:
                bandwidth = float(line.split('Mbits/sec')[0].split()[-1])
                break
            elif 'Gbits/sec' in line and 'sec' in line.split('Gbits/sec')[0]:
                bandwidth = float(line.split('Gbits/sec')[0].split()[-1]) * 1000  
                break
        
        source.cmd('kill ' + ping_pid.strip())
        
        ping_result = source.cmd('cat /tmp/ping_results_{0}'.format(pair_id))
        source.cmd('rm /tmp/ping_results_{0}'.format(pair_id))
        
        avg_latency = "Unknown"
        packet_loss = "Unknown"
        
        for line in ping_result.split('\n'):
            if 'min/avg/max' in line:
                avg_latency = float(line.split('=')[1].split('/')[1].strip())
            if 'packet loss' in line:
                packet_loss = line.split(',')[2].strip()
        
        results[pair_id] = {
            'source': source.name,
            'dest': dest.name,
            'bandwidth': bandwidth,
            'latency': avg_latency,
            'packet_loss': packet_loss
        }
    
    # Start all threads
    for i, (source, dest) in enumerate(host_pairs):
        thread = Thread(target=run_iperf_client, args=(source, dest, i))
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    print("\n*** All traffic tests completed")
    
    print("\n*** SIMULTANEOUS TRAFFIC TEST RESULTS ***")
    
    total_bandwidth = 0
    total_latency = 0
    total_loss = 0
    
    for i in range(len(host_pairs)):
        result = results[i]
        
        bandwidth = result['bandwidth']
        avg_latency = result['latency']
        packet_loss = result['packet_loss']
        
        loss_value = 0
        if packet_loss != "Unknown":
            loss_value = float(packet_loss.split('%')[0])
        
        total_bandwidth += bandwidth
        
        if avg_latency != "Unknown":
            total_latency += avg_latency
            
        total_loss += loss_value
        
        print("Pair {0}: {1} -> {2}".format(i+1, result['source'], result['dest']))
        print("  - Bandwidth: {0} Mbits/sec".format(bandwidth))
        print("  - Latency under load: {0} ms".format(avg_latency))
        print("  - Packet loss: {0}".format(packet_loss))

    for _, dest in host_pairs:
        dest.cmd('kill %iperf')
    
    print("\n*** test results")
    print(" average packet loss: {0}%".format(total_loss/len(host_pairs)))
    print(" average latency: {0} ms".format(total_latency/len(host_pairs)))
    print(" average bandwidth: {0} Mbits/sec".format(total_bandwidth/len(host_pairs)))

def test_fault_tolerance(net, num_failures=1):
    print("\n*** Testing network fault tolerance with {0} link failure(s) ***\n".format(num_failures))

    all_links = net.links
    
    host_pairs = []
    
    for i in range(0, 10, 2):
        host_pairs.append((net.get('h{0}'.format(i + 1)), net.get('h{0}'.format(10 - i))))
    
    print("*** Running baseline performance tests")
    baseline_results = {}
    
    for i, (source, dest) in enumerate(host_pairs):
        print("Pair {0}: Testing baseline performance from {1} to {2}".format(
            i+1, source.name, dest.name))
        
        # Test connectivity
        ping_result = source.cmd('ping -c 3 -W 1 {0}'.format(dest.IP()))
        connectivity = not ('100% packet loss' in ping_result)
        
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
            if 'Mbits/sec' in line:
                bandwidth = float(line.split('Mbits/sec')[0].split()[-1])
                break
            elif 'Gbits/sec' in line:
                bandwidth = float(line.split('Gbits/sec')[0].split()[-1]) * 1000 
                break
        
        
        baseline_results[i] = {
            'connectivity': connectivity,
            'latency': latency,
            'bandwidth': bandwidth,
        }
    
    # Function to fail links and test impact
    def test_with_failed_links(links_to_fail):
        for link in links_to_fail:
            src, dst = link.intf1.node, link.intf2.node
            print("*** Failing link between {0} and {1}".format(
                src.name, dst.name))
            
            src.cmd('ifconfig {0} down'.format(link.intf1.name))
            dst.cmd('ifconfig {0} down'.format(link.intf2.name))
        
        print("*** Waiting for network to reconverge")
        time.sleep(5)
        
        failure_results = {}
        
        for i, (source, dest) in enumerate(host_pairs):
            print("Pair {0}: Testing performance from {1} to {2} with failed links".format(
                i+1, source.name, dest.name))
            
            # Test connectivity
            ping_result = source.cmd('ping -c 3 -W 1 {0}'.format(dest.IP()))
            connectivity = not ('100% packet loss' in ping_result)
            
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
                    if 'Mbits/sec' in line:
                        bandwidth = float(line.split('Mbits/sec')[0].split()[-1])
                        break
                    elif 'Gbits/sec' in line:
                        bandwidth = float(line.split('Gbits/sec')[0].split()[-1]) * 1000 
                        break
                  
            
            failure_results[i] = {
                'connectivity': connectivity,
                'latency': latency,
                'bandwidth': bandwidth
            }
        
        return failure_results
    
    router_links = [link for link in all_links 
                   if (hasattr(link.intf1.node, 'name') and 
                       hasattr(link.intf2.node, 'name') and
                       link.intf1.node.name.startswith('r') and 
                       link.intf2.node.name.startswith('r'))]
    links_to_test = router_links if router_links else all_links
    num_failures = min(num_failures, len(links_to_test))
    
    failed_links = random.sample(links_to_test, num_failures)
    failure_results = test_with_failed_links(failed_links)
    
    # Restore network
    print("\n*** Restoring network links")
    for link in failed_links:
        src, dst = link.intf1.node, link.intf2.node
        src.cmd('ifconfig {0} up'.format(link.intf1.name))
        dst.cmd('ifconfig {0} up'.format(link.intf2.name))
    
    print("*** Waiting for network to recover")
    time.sleep(5)
    
    failed_links_str = ", ".join(["{0}-{1}".format(link.intf1.node.name, link.intf2.node.name) 
                                for link in failed_links])
    
    # Print comparison results
    print("\n*** FAULT TOLERANCE TEST RESULTS ***")
    print("Failed links: {0}".format(failed_links_str))
    
    for i, (source, dest) in enumerate(host_pairs):
        print("\nPair {0}: {1} -> {2}".format(i+1, source.name, dest.name))
        
        baseline = baseline_results[i]
        failure = failure_results[i]
        
        print("  Connectivity before failure: {0}".format("YES" if baseline['connectivity'] else "NO"))
        print("  Connectivity after failure: {0}".format("YES" if failure['connectivity'] else "NO"))
        
        if baseline['connectivity'] and failure['connectivity']:
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
            
            try:
                baseline_bw = float(baseline['bandwidth']) if isinstance(baseline['bandwidth'], str) else baseline['bandwidth']
                failure_bw = float(failure['bandwidth']) if isinstance(failure['bandwidth'], str) else failure['bandwidth']
                
                if baseline_bw != "N/A" and failure_bw != "N/A":
                    bw_change = ((failure_bw - baseline_bw) / baseline_bw) * 100
                    print("  Bandwidth before: {0}".format(baseline['bandwidth']))
                    print("  Bandwidth after: {0} ({1:+.1f}%)".format(
                        failure['bandwidth'], bw_change))
                else:
                    print("  Bandwidth before: {0}".format(baseline['bandwidth']))
                    print("  Bandwidth after: {0}".format(failure['bandwidth']))
            except (ValueError, TypeError, ZeroDivisionError):
                print("  Bandwidth before: {0}".format(baseline['bandwidth']))
                print("  Bandwidth after: {0}".format(failure['bandwidth']))

        if not baseline['connectivity']:
            print("  ASSESSMENT: Pair was disconnected before link failure")
        elif not failure['connectivity']:
            print("  ASSESSMENT: FAILED - Connection lost after link failure")
        else:
            try:
                baseline_latency = float(baseline['latency'].split()[0])
                failure_latency = float(failure['latency'].split()[0])
                latency_change = ((failure_latency - baseline_latency) / baseline_latency) * 100
                
                if latency_change > 100:  
                    print("  ASSESSMENT: DEGRADED - Significant performance impact")
                elif latency_change > 20: 
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
        
        print('*** printing host connections\n')
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