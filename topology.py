from mininet.net import Mininet
from mininet.node import Controller, RemoteController
from mininet.node import Host
from mininet.node import OVSKernelSwitch
from mininet.cli import CLI
from mininet.log import setLogLevel
from mininet.link import TCLink

def topology():
    net = Mininet(topo=None, build=False, ipBase='10.0.0.0/8', link=TCLink)

    # Add controller
    print('*** Adding controller')
    c0 = net.addController(name='c0', controller=RemoteController, ip='127.0.0.1', port=6633)

    num = 5

    # Add switches (these will handle both L2 and L3 via the controller)
    switches = []
    print('*** Adding switches')
    for i in range(num):
        switch_name = 's{0}'.format(i+1)
        switch = net.addSwitch(switch_name, cls=OVSKernelSwitch, protocols='OpenFlow13')
        switches.append(switch)

    # Add hosts with proper subnetting
    print('*** Adding hosts')
    hosts = []
    for i in range(num):
        for j in range(2):
            host_index = i*2 + j + 1
            host_name = 'h{0}'.format(host_index)
            # Set IP address with subnet
            host = net.addHost(host_name, cls=Host, 
                              ip='10.0.{0}.{1}/24'.format(i+1, 100+j), 
                              defaultRoute='via 10.0.{0}.1'.format(i+1))
            hosts.append(host)
            # Connect hosts to their switch
            net.addLink(host, switches[i])
            
    # Add gateway IPs to switches (the controller will handle these virtual IPs)
    print('*** Setting up gateway IPs for the controller to manage')
    for i in range(num):
        # No actual IP configuration here - the controller will handle it
        print(f"Switch s{i+1} will act as gateway for subnet 10.0.{i+1}.0/24")

    # Create mesh topology by connecting all switches to each other
    print('*** Creating mesh links between switches')
    for i in range(num):
        for j in range(i+1, num):
            print(f"Creating link between s{i+1} and s{j+1}")
            net.addLink(switches[i], switches[j])

    # Start network
    print('*** Starting network')
    net.build()

    # Start controller
    c0.start()

    # Start switches
    for switch in switches:
        switch.start([c0])

    # Add ARP entries for gateway IPs to all hosts
    print('*** Adding static ARP entries for gateways')
    for i in range(num):
        for j in range(2):
            host_index = i*2 + j
            if host_index < len(hosts):
                # Create a fake MAC address for gateway
                gateway_mac = '00:00:00:00:{0:02x}:01'.format(i+1)
                gateway_ip = '10.0.{0}.1'.format(i+1)
                hosts[host_index].cmd(f'arp -s {gateway_ip} {gateway_mac}')
                print(f"Added static ARP for {hosts[host_index].name}: {gateway_ip} -> {gateway_mac}")

    # Add static routes for cross-subnet communication
    print('*** Adding static routes for cross-subnet communication')
    for i, host in enumerate(hosts):
        host_subnet = i // 2 + 1
        for target_subnet in range(1, num+1):
            if target_subnet != host_subnet:
                host.cmd(f'route add -net 10.0.{target_subnet}.0/24 gw 10.0.{host_subnet}.1')
                print(f"Added route on {host.name}: 10.0.{target_subnet}.0/24 via 10.0.{host_subnet}.1")

    return net

if __name__ == '__main__':
    setLogLevel('info')  
    net = topology()
    CLI(net)
    net.stop()