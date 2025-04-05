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

def test_simultaneous_traffic(net, num_pairs=5, duration=10):
    print("\n*** Testing simultaneous network traffic\n")
    
    all_hosts = [h for h in net.hosts if h.name.startswith('h')]
    
    if len(all_hosts) < num_pairs * 2:
        print("WARNING: Not enough hosts for {0} pairs. Using {1} pairs instead.".format(
            num_pairs, len(all_hosts) // 2))
        num_pairs = len(all_hosts) // 2
    
    host_pairs = []
    hosts_copy = all_hosts[:] 
    random.shuffle(hosts_copy)
    
    for i in range(num_pairs):
        if len(hosts_copy) >= 2:
            source = hosts_copy.pop()
            dest = hosts_copy.pop()
            host_pairs.append((source, dest))
    
    # Start iperf servers with larger window size for better throughput
    for _, dest in host_pairs:
        dest.cmd('iperf -s -w 256k &')
    
    print("*** Started {0} iperf servers".format(len(host_pairs)))
    time.sleep(2)  # Increased sleep for server stabilization
    
    threads = []
    results = {}
    bandwidth_values = []
    retrans_counts = []
    
    def run_iperf_client(source, dest, pair_id):
        print("*** Starting traffic from {0} to {1}".format(source.name, dest.name))
        # Run iperf with parallel streams and increased window size
        iperf_cmd = 'iperf -c {0} -t {1} -i 1 -P 4 -w 256k'.format(dest.IP(), duration)
        iperf_result = source.cmd(iperf_cmd)
        results[pair_id] = iperf_result
        
        # Extract numeric bandwidth value (in Mbits/sec)
        bw = 0.0
        retrans = 0
        for line in iperf_result.split('\n'):
            if 'Mbits/sec' in line and 'SUM' not in line:
                bw = float(line.split()[6])
                break
            elif 'Gbits/sec' in line and 'SUM' not in line:
                bw = float(line.split()[6]) * 1000  # Convert to Mbits/sec
                break
            elif 'retransmits' in line:
                retrans = int(line.split()[7])
        
        bandwidth_values.append(bw)
        retrans_counts.append(retrans)
        
        # Run detailed ping test during traffic
        ping_result = source.cmd('ping -c 5 -i 0.2 -W 1 {0}'.format(dest.IP()))
        results[pair_id] = (iperf_result, ping_result)
    
    # Start all client threads with timeout handling
    for i, (source, dest) in enumerate(host_pairs):
        thread = Thread(target=run_iperf_client, args=(source, dest, i))
        threads.append(thread)
        thread.start()
        time.sleep(0.5)  # Stagger thread starts
    
    # Wait for threads with timeout
    start_time = time.time()
    for thread in threads:
        remaining_time = max(0, (duration + 10) - (time.time() - start_time))
        thread.join(remaining_time)
        if thread.is_alive():
            print("WARNING: Thread timed out")
    
    print("\n*** All traffic tests completed")
    
    print("\n*** SIMULTANEOUS TRAFFIC TEST RESULTS ***")
    
    total_bandwidth = sum(bandwidth_values) if bandwidth_values else 0
    total_retrans = sum(retrans_counts) if retrans_counts else 0
    
    for i, (source, dest) in enumerate(host_pairs):
        iperf_result, ping_result = results.get(i, ("", ""))
        
        # Bandwidth display
        bandwidth = "Unknown"
        for line in iperf_result.split('\n'):
            if 'Mbits/sec' in line and 'SUM' not in line:
                bandwidth = "{0:.2f} Mbits/sec".format(float(line.split()[6]))
                break
            elif 'Gbits/sec' in line and 'SUM' not in line:
                bandwidth = "{0:.2f} Gbits/sec".format(float(line.split()[6]))
                break
        
        # Ping statistics
        avg_latency = "Unknown"
        packet_loss = "Unknown"
        jitter = "Unknown"
        
        for line in ping_result.split('\n'):
            if 'min/avg/max/mdev' in line:
                parts = line.split('=')[1].split('/')
                avg_latency = "{0} ms".format(parts[1].strip())
                jitter = "{0} ms".format(parts[3].strip())
            elif 'packet loss' in line:
                packet_loss = line.split(',')[2].strip()
        
        # Retransmission count
        retrans = 0
        for line in iperf_result.split('\n'):
            if 'retransmits' in line:
                retrans = int(line.split()[7])
                break
        
        print("Pair {0}: {1} -> {2}".format(i+1, source.name, dest.name))
        print("  - Bandwidth: {0}".format(bandwidth))
        print("  - Latency under load: {0}".format(avg_latency))
        print("  - Packet loss: {0}".format(packet_loss))
        print("  - Jitter: {0}".format(jitter))
        print("  - TCP retransmits: {0}".format(retrans))
    
    # Clean up servers
    for _, dest in host_pairs:
        dest.cmd('kill %iperf')
    
    print("\n*** Aggregate statistics:")
    print("  - Total bandwidth: {0:.2f} Mbits/sec".format(total_bandwidth))
    print("  - Average bandwidth per pair: {0:.2f} Mbits/sec".format(
        total_bandwidth/len(host_pairs) if host_pairs else 0))
    print("  - Total TCP retransmissions: {0}".format(total_retrans))
    print("  - Average retransmissions per pair: {0:.1f}".format(
        total_retrans/len(host_pairs) if host_pairs else 0))
    print("\n*** Testing simultaneous network traffic\n")
    
    all_hosts = [h for h in net.hosts if h.name.startswith('h')]
    
    if len(all_hosts) < num_pairs * 2:
        print("WARNING: Not enough hosts for {0} pairs. Using {1} pairs instead.".format(
            num_pairs, len(all_hosts) // 2))
        num_pairs = len(all_hosts) // 2
    
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
    for i, (source, dest) in enumerate(host_pairs):
        iperf_result = results[i]
        
        bandwidth = "Unknown"
        for line in iperf_result.split('\n'):
            if 'Mbits/sec' in line:
                bandwidth = line.split('Mbits/sec')[0].split()[-1] + " Mbits/sec"
                total_bandwidth += bandwidth
                break
            elif 'Gbits/sec' in line:
                bandwidth = line.split('Gbits/sec')[0].split()[-1] + " Gbits/sec"
                total_bandwidth += bandwidth
                break
        
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
    
    for _, dest in host_pairs:
        dest.cmd('kill %iperf')
    
    print("\n*** Aggregate network bandwidth: {0:.2f} Mbits/sec".format(total_bandwidth))

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