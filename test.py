from mininet.net import Mininet
from mininet.node import Node, Controller
from mininet.cli import CLI
from mininet.log import setLogLevel, info
import time
from topology import MyTopo  

def run_tests(net):
    """Run network tests after starting the topology."""
    info("\n*** Running network tests ***\n")

    # Get host and router references
    h1, h2, h3, h4 = net.get('h1', 'h2', 'h3', 'h4')
    r1, r2 = net.get('r1', 'r2')

    # === 1. Basic Connectivity Test (Ping between all hosts) ===
    info("\n*** Testing Host Connectivity (Ping) ***\n")
    hosts = [h1, h2, h3, h4]
    for src in hosts:
        for dst in hosts:
            if src != dst:
                result = src.cmd('ping -c 3 ' + dst.IP())
                if "0% packet loss" in result:
                    info("[PASS] " + src.name + " can reach " + dst.name + "\n")
                else:
                    info("[FAIL] " + src.name + " cannot reach " + dst.name + "\n")

    # === 2. Check Routing Tables on Routers ===
    info("\n*** Checking Router Routing Tables ***\n")
    r1_routes = r1.cmd('ip route')
    r2_routes = r2.cmd('ip route')
    info("Routing Table of r1:\n" + r1_routes + "\n")
    info("Routing Table of r2:\n" + r2_routes + "\n")

    # === 3. Latency Check (Ping with timing) ===
    info("\n*** Measuring Network Latency ***\n")
    latency = h1.cmd('ping -c 5 ' + h3.IP())
    info("Latency Test from h1 to h3:\n" + latency + "\n")

    # === 4. Bandwidth Test using iperf ===
    info("\n*** Measuring Network Bandwidth ***\n")
    h3.cmd('iperf -s -D')  # Start iperf server on h3 in daemon mode
    time.sleep(1)  # Give server time to start
    bandwidth = h1.cmd('iperf -c ' + h3.IP() + ' -t 5')
    info("Bandwidth Test from h1 to h3:\n" + bandwidth + "\n")

    # === 5. Check if IP Forwarding is Enabled on Routers ===
    info("\n*** Checking IP Forwarding on Routers ***\n")
    for router in [r1, r2]:
        ip_forward_status = router.cmd("cat /proc/sys/net/ipv4/ip_forward").strip()
        if ip_forward_status == "1":
            info("[PASS] IP forwarding is enabled on " + router.name + "\n")
        else:
            info("[FAIL] IP forwarding is NOT enabled on " + router.name + "\n")

    info("\n*** Network Tests Completed ***\n")

def main():
    setLogLevel('info')

    info("\n*** Creating Network ***\n")
    net = Mininet(topo=MyTopo(), controller=Controller)
    net.start()

    try:
        run_tests(net)
    finally:
        info("\n*** Stopping Network ***\n")
        net.stop()

if __name__ == '__main__':
    main()