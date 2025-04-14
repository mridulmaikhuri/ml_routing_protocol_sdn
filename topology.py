from mininet.net import Mininet
from mininet.node import RemoteController
from mininet.node import Host, Node
from mininet.node import OVSKernelSwitch
from mininet.cli import CLI
from mininet.log import setLogLevel

def topology():
    # Initialize Mininet
    net = Mininet(topo=None, build=False, ipBase='10.0.0.0/8')

    # Add controller
    print('*** Adding controller')
    c0 = net.addController(name='c0', 
                         controller=RemoteController,  # Using RemoteController instead
                         protocol='tcp', 
                         port=6633,
                         ip='127.0.0.1') 

    # Add Host
    h1 = net.addHost('h1')
    h2 = net.addHost('h2')
    h3 = net.addHost('h3')
    h4 = net.addHost('h4')

    s1 = net.addSwitch('s1', cls = OVSKernelSwitch)
    s2 = net.addSwitch('s2', cls = OVSKernelSwitch)
    s3 = net.addSwitch('s3', cls = OVSKernelSwitch)
    s4 = net.addSwitch('s4', cls = OVSKernelSwitch)

    net.addLink(h1, s1)
    net.addLink(h2, s1)
    net.addLink(s1, s2)
    net.addLink(s1, s3)
    net.addLink(s2, s4)
    net.addLink(s3, s4)
    net.addLink(s4, h3)
    net.addLink(s4, h4)

    # Start network
    print('*** Starting network')
    net.build()

    # Start switches
    for switch in [s1, s2, s3, s4]:
        switch.start([c0])

    return net

if __name__ == '__main__':
    setLogLevel('info')  
    net = topology()
    CLI(net)
    net.stop()