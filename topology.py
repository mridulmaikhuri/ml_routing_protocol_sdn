from mininet.net import Mininet
from mininet.node import Controller
from mininet.node import Host, Node
from mininet.node import OVSKernelSwitch
from mininet.cli import CLI
from mininet.log import setLogLevel

def topology():
    # Initialize Mininet
    net = Mininet(topo=None, build=False, ipBase='10.0.0.0/8')

    # Add controller
    print('*** Adding controller')
    c0 = net.addController(name='c0', controller=Controller, protocol='tcp', port=6633)

    # Define number of routers and switches
    num = 5

    # Add routers
    routers = []
    print('*** Adding routers')
    for i in range(num):
        router_name = 'r{0}'.format(i+1)
        router = net.addHost(router_name, cls=Node, ip='10.0.{0}.1/24'.format(i+1))
        router.cmd('sysctl -w net.ipv4.ip_forward=1')       
        routers.append(router)

    # Add switches
    switches = []
    print('*** Adding switches')
    for i in range(num):
        switch_name = 's{0}'.format(i+1)
        switch = net.addSwitch(switch_name, cls=OVSKernelSwitch)
        switches.append(switch)

    # Add hosts 
    print('** Adding hosts')
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
    print('*** Creating links between switches and routers')
    for i in range(num):
        net.addLink(switches[i], routers[i], 
                   intfName2='r{0}-eth0'.format(i+1), 
                   params2={'ip': '10.0.{0}.1/24'.format(i+1)})

    # Create mesh topology by connecting all routers to each other
    print('*** Creating mesh links between routers')
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
    print('*** Starting network')
    net.build()

    # Start controller
    c0.start()

    # Start switches
    for switch in switches:
        switch.start([c0])

    return net

if __name__ == '__main__':
    setLogLevel('info')  
    net = topology()
    CLI(net)
    net.stop()