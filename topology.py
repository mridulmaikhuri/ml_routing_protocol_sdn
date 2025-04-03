from mininet.net import Mininet
from mininet.node import Controller, RemoteController, OVSController
from mininet.node import CPULimitedHost, Host, Node
from mininet.node import OVSKernelSwitch, UserSwitch
from mininet.node import IVSSwitch
from mininet.cli import CLI
from mininet.log import setLogLevel, info
from mininet.link import TCLink, Intf
from subprocess import call

def meshNetworkWithRouters():
    # Initialize Mininet
    net = Mininet(topo=None, build=False, ipBase='10.0.0.0/8')
    
    # Add controller
    info('*** Adding controller\n')
    c0 = net.addController(name='c0', controller=Controller, protocol='tcp', port=6633)
    
    # Number of routers for the mesh
    num_routers = 5
    routers = []
    
    # Add routers
    info('*** Adding routers\n')
    for i in range(num_routers):
        router_name = 'r{0}'.format(i+1)
        router = net.addHost(router_name, cls=Node, ip='10.0.{0}.1/24'.format(i+1))
        router.cmd('sysctl -w net.ipv4.ip_forward=1')  # Enable IP forwarding
        routers.append(router)
    
    # Add hosts (one per router)
    info('*** Adding hosts\n')
    hosts = []
    for i in range(num_routers):
        host_name = 'h{0}'.format(i+1)
        host = net.addHost(host_name, cls=Host, 
                         ip='10.0.{0}.100/24'.format(i+1), 
                         defaultRoute='via 10.0.{0}.1'.format(i+1))
        hosts.append(host)
    
    # Connect hosts to their respective routers
    info('*** Creating links between hosts and routers\n')
    for i in range(num_routers):
        net.addLink(hosts[i], routers[i], 
                   intfName2='r{0}-eth0'.format(i+1), 
                   params2={'ip': '10.0.{0}.1/24'.format(i+1)})
    
    # Create mesh topology by connecting all routers to each other
    info('*** Creating mesh links between routers\n')
    link_count = 0
    for i in range(num_routers):
        for j in range(i+1, num_routers):
            link_count += 1
            # Create subnet for each router-router link
            subnet = 10 + link_count  # Start from subnet 10.0.11.0/24 and increment
            
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
    
    # Run CLI
    info('*** Running CLI\n')
    CLI(net)
    
    # Stop network
    info('*** Stopping network\n')
    net.stop()

if __name__ == '__main__':
    setLogLevel('info')
    meshNetworkWithRouters()