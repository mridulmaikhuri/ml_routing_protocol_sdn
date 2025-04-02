from mininet.net import Mininet
from mininet.topo import Topo
from mininet.node import Controller, OVSSwitch, Node
from mininet.cli import CLI
from mininet.log import setLogLevel

class MyTopo(Topo):
    def build(self):
        # Create routers
        r1 = self.addNode('r1', cls=Node, ip='192.168.1.1/24')
        r2 = self.addNode('r2', cls=Node, ip='192.168.1.2/24')
        r3 = self.addNode('r3', cls=Node, ip='192.168.1.3/24')
        r4 = self.addNode('r4', cls=Node, ip='192.168.1.4/24')

        # Create switch
        s1 = self.addSwitch('s1')
        s2 = self.addSwitch('s2')
        s3 = self.addSwitch('s3')
        s4 = self.addSwitch('s4')

        # Add hosts
        h1 = self.addHost('h1', ip='10.0.0.1/24', defaultRoute='via 10.0.0.3')
        h2 = self.addHost('h2', ip='10.0.0.2/24', defaultRoute='via 10.0.0.3')

        h3 = self.addHost('h3', ip='10.0.1.1/24', defaultRoute='via 10.0.1.3')
        h4 = self.addHost('h4', ip='10.0.1.2/24', defaultRoute='via 10.0.1.3')

        h5 = self.addHost('h5', ip='10.0.2.1/24', defaultRoute='via 10.0.2.3')
        h6 = self.addHost('h6', ip='10.0.2.2/24', defaultRoute='via 10.0.2.3')

        h7 = self.addHost('h7', ip='10.0.3.1/24', defaultRoute='via 10.0.3.3')
        h8 = self.addHost('h8', ip='10.0.3.2/24', defaultRoute='via 10.0.3.3')

        # Connect hosts to switch
        self.addLink(h1, s1)
        self.addLink(h2, s1)

        self.addLink(h3, s2)
        self.addLink(h4, s2)

        self.addLink(h5, s3)
        self.addLink(h6, s3)

        self.addLink(h7, s4)
        self.addLink(h8, s4)

        # Connect switch to routers
        self.addLink(s1, r1, intfName1='r1-eth1', params1={'ip': '10.0.0.3/24'})
        self.addLink(s1, r3, intfName1='r3-eth1', params1={'ip': '10.0.0.4/24'})

        self.addLink(s4, r1, intfName1='r1-eth2', params1={'ip': '10.0.3.3/24'})
        self.addLink(s4, r2, intfName1='r2-eth1', params1={'ip': '10.0.3.4/24'})

        self.addLink(s3, r2, intfName1='r2-eth2', params1={'ip': '10.0.2.3/24'})
        self.addLink(s3, r4, intfName1='r4-eth1', params1={'ip': '10.0.2.4/24'})

        self.addLink(s2, r4, intfName1='r4-eth2', params1={'ip': '10.0.1.3/24'})
        self.addLink(s2, r3, intfName1='r3-eth2', params1={'ip': '10.0.1.4/24'})

        # Connect routers to each other
        self.addLink(r1, r2, intfName1='r1-eth3', params1={'ip': '192.168.2.1/30'}, intfName2='r2-eth3', params2={'ip': '192.168.2.2/30'})
        self.addLink(r3, r4, intfName1='r3-eth3', params1={'ip': '192.168.3.1/30'}, intfName2='r4-eth3', params2={'ip': '192.168.3.2/30'})
        self.addLink(r1, r3, intfName1='r1-eth4', params1={'ip': '192.168.4.1/30'}, intfName2='r3-eth4', params2={'ip': '192.168.4.2/30'})
        self.addLink(r2, r4, intfName1='r2-eth4', params1={'ip': '192.168.5.1/30'}, intfName2='r4-eth4', params2={'ip': '192.168.5.2/30'})
        self.addLink(r1, r4, intfName1='r1-eth5', params1={'ip': '192.168.6.1/30'}, intfName2='r4-eth5', params2={'ip': '192.168.6.2/30'})
        self.addLink(r2, r3, intfName1='r2-eth5', params1={'ip': '192.168.7.1/30'}, intfName2='r3-eth5', params2={'ip': '192.168.7.2/30'})


def run():
    net = Mininet(topo=MyTopo(), controller=Controller)
    net.start()

    r1 = net.get('r1')
    r2 = net.get('r2')
    r3 = net.get('r3')
    r4 = net.get('r4')

    # Enable Ip forwarding on routers
    for r in [r1, r2, r3, r4]:
        r.cmd('sysctl -w net.ipv4.ip_forward=1')

    # add static routes to routers
    r1.cmd('ip route add 10.0.1.0/24 via 192.168.4.2')  # Reach h3, h4 via r3
    r1.cmd('ip route add 10.0.2.0/24 via 192.168.6.2')  # Reach h5, h6 via r4
    r1.cmd('ip route add 10.0.3.0/24 via 10.0.3.4')  # Directly connected via eth2
    r1.cmd('ip route add 192.168.5.0/30 via 192.168.2.2')  # Reach r2 via r1-r2 link
    r1.cmd('ip route add 10.0.2.0/24 via 192.168.2.2')  # Reach h5, h6 via r2
    r1.cmd('ip route add 10.0.1.0/24 via 192.168.2.2')  # Reach h3, h4 via r2-r4

    r2.cmd('ip route add 10.0.0.0/24 via 192.168.2.1')  # Reach h1, h2 via r1
    r2.cmd('ip route add 10.0.1.0/24 via 192.168.5.2')  # Reach h3, h4 via r4
    r2.cmd('ip route add 10.0.3.0/24 via 10.0.3.3')  # Directly connected

    r3.cmd('ip route add 10.0.0.0/24 via 192.168.4.1')  # Reach h1, h2 via r1
    r3.cmd('ip route add 10.0.2.0/24 via 192.168.3.2')  # Reach h5, h6 via r4
    r3.cmd('ip route add 10.0.3.0/24 via 192.168.4.1')  # Via r1

    r4.cmd('ip route add 10.0.0.0/24 via 192.168.6.1')  # Reach h1, h2 via r1
    r4.cmd('ip route add 10.0.3.0/24 via 192.168.5.1')  # Reach h7, h8 via r2
    r4.cmd('ip route add 10.0.0.0/24 via 192.168.3.1')  # Reach via r3
    
    CLI(net)
    net.stop()

if __name__ == '__main__':
    setLogLevel('info')
    run()