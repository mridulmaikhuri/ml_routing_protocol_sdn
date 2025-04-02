from mininet.net import Mininet
from mininet.topo import Topo
from mininet.node import Controller, Node
from mininet.cli import CLI
from mininet.log import setLogLevel

class MyTopo(Topo):
    def build(self):
        # Create a router (Linux host acting as a router)
        router = self.addNode('r1', ip='192.168.1.1/24')

        # Create two switches for two subnets
        switch1 = self.addSwitch('s1')
        switch2 = self.addSwitch('s2')

        # Add hosts in different subnets
        h1 = self.addHost('h1', ip='192.168.1.2/24', defaultRoute='via 192.168.1.1')
        h2 = self.addHost('h2', ip='192.168.1.3/24', defaultRoute='via 192.168.1.1')
        h3 = self.addHost('h3', ip='10.0.0.2/24', defaultRoute='via 10.0.0.1')
        h4 = self.addHost('h4', ip='10.0.0.3/24', defaultRoute='via 10.0.0.1')

        # Connect hosts to respective switches
        self.addLink(h1, switch1)
        self.addLink(h2, switch1)
        self.addLink(h3, switch2)
        self.addLink(h4, switch2)

        # Connect router to both switches
        self.addLink(router, switch1, intfName1='r1-eth1', params1={'ip': '192.168.1.1/24'})
        self.addLink(router, switch2, intfName1='r1-eth2', params1={'ip': '10.0.0.1/24'})

def run():
    net = Mininet(topo=MyTopo(), controller=Controller)
    net.start()

    router = net.get('r1')

    # Enable IP forwarding on the router
    router.cmd('sysctl -w net.ipv4.ip_forward=1')

    CLI(net)
    net.stop()

if __name__ == '__main__':
    setLogLevel('info')
    run()