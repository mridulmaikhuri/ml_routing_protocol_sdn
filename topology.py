
from mininet.net import Mininet
from mininet.topo import Topo
from mininet.node import Node, Controller
from mininet.cli import CLI
from mininet.log import setLogLevel, info

class MyTopo(Topo):
    def build(self):
        # Create router nodes without a default IP to avoid conflicts.
        r1 = self.addNode('r1', cls=Node)
        r2 = self.addNode('r2', cls=Node)
        
        # Create switches.
        s1 = self.addSwitch('s1')
        s2 = self.addSwitch('s2')
        
        # Create hosts with default routes pointing to their respective router interfaces.
        h1 = self.addHost('h1', ip='10.0.0.1/24', defaultRoute='via 10.0.0.254')
        h2 = self.addHost('h2', ip='10.0.0.2/24', defaultRoute='via 10.0.0.254')
        h3 = self.addHost('h3', ip='10.0.1.1/24', defaultRoute='via 10.0.1.254')
        h4 = self.addHost('h4', ip='10.0.1.2/24', defaultRoute='via 10.0.1.254')
        
        # Connect hosts to their switches.
        self.addLink(h1, s1)
        self.addLink(h2, s1)
        self.addLink(h3, s2)
        self.addLink(h4, s2)
        
        # Connect switches to routers.
        # s1 to r1: r1's interface gets IP 10.0.0.254/24
        self.addLink(s1, r1, intfName2='r1-eth1', params2={'ip': '10.0.0.254/24'})
        # s2 to r2: r2's interface gets IP 10.0.1.254/24
        self.addLink(s2, r2, intfName2='r2-eth1', params2={'ip': '10.0.1.254/24'})
        
        # Connect routers r1 and r2 using a separate subnet (192.168.1.0/24).
        self.addLink(r1, r2,
                     intfName1='r1-eth2', params1={'ip': '192.168.1.1/24'},
                     intfName2='r2-eth2', params2={'ip': '192.168.1.2/24'})

def run():
    topo = MyTopo()
    net = Mininet(topo=topo, controller=Controller)
    net.start()
    
    # Get router nodes.
    r1 = net.get('r1')
    r2 = net.get('r2')
    
    # Enable IP forwarding on the routers.
    for router in [r1, r2]:
        router.cmd('sysctl -w net.ipv4.ip_forward=1')
    
    # Configure static routes:
    # r1: route to 10.0.1.0/24 via r2's router-to-router link.
    r1.cmd('ip route add 10.0.1.0/24 via 192.168.1.2 dev r1-eth2')
    # r2: route to 10.0.0.0/24 via r1's router-to-router link.
    r2.cmd('ip route add 10.0.0.0/24 via 192.168.1.1 dev r2-eth2')
    
    info('*** Running CLI\n')
    CLI(net)
    net.stop()

if __name__ == '__main__':
    setLogLevel('info')
    run()
