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
            print(" PASS: All required routes are present in routing table")

def test_simultaneous_traffic(net, num_pairs=3, duration=10):  
    print("\n*** Testing simultaneous network traffic between {0} host pairs for {1} seconds\n".format(num_pairs, duration))
    
    # Get all hosts and filter out routers
    all_hosts = [h for h in net.hosts if h.name.startswith('h')]
    
    if len(all_hosts) < num_pairs * 2:
        print("WARNING: Not enough hosts for {0} pairs. Using {1} pairs instead.".format(
            num_pairs, len(all_hosts) // 2))
        num_pairs = len(all_hosts) // 2
    
    # Select random pairs of hosts
    host_pairs = []
    hosts_copy = all_hosts[:]  # Create a copy without using .copy() method
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
    total_latency = 0
    total_loss = 0
    for i, (source, dest) in enumerate(host_pairs):
        iperf_result = results[i]
        
        # Extract bandwidth from results
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
                total_packet_loss += float(packet_loss.split('%')[0])
        
        print("Pair {0}: {1} -> {2}".format(i+1, source.name, dest.name))
        print("  - Bandwidth: {0}".format(bandwidth))
        print("  - Latency under load: {0} ms".format(avg_latency))
        print("  - Packet loss: {0}".format(packet_loss))

    for _, dest in host_pairs:
        dest.cmd('kill %iperf')
    
    print("\n*** test results\n")
    print(" average packet loss: {0}".format(packet_loss/num_pairs))
    print(" average latency: {0}".format(total_latency/num_pairs))
    print(" average bandwidth: {0}".format(total_bandwidth/num_pairs))
    

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
        
        check_routing_tables(net)   
        test_connectivity(net)
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