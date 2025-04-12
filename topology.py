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

    # Add switches
    switches = []
    print('*** Adding switches')
    for i in range(num):
        switch_name = 's{0}'.format(i+1)
        switch = net.addSwitch(switch_name, cls=OVSKernelSwitch)
        switches.append(switch)
    
    # Add backbone switches (replacing routers)
    backbone_switches = []
    print('*** Adding backbone switches')
    for i in range(num):
        bs_name = 'bs{0}'.format(i+1)
        bs = net.addSwitch(bs_name, cls=OVSKernelSwitch)
        backbone_switches.append(bs)

    # Add hosts 
    print('*** Adding hosts')
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

    # Connect access switches to their respective backbone switches
    print('*** Creating links between access and backbone switches')
    for i in range(num):
        net.addLink(switches[i], backbone_switches[i])

    # Create mesh topology by connecting all backbone switches to each other
    print('*** Creating mesh links between backbone switches')
    for i in range(num):
        for j in range(i+1, num):
            net.addLink(backbone_switches[i], backbone_switches[j])

    # Start network
    print('*** Starting network')
    net.build()

    # Start controller
    c0.start()

    # Start all switches
    for switch in switches + backbone_switches:
        switch.start([c0])

    return net

if __name__ == '__main__':
    setLogLevel('info')  
    net = topology()
    CLI(net)
    net.stop()