from mininet.net import Mininet
from mininet.topo import Topo
from mininet.cli import CLI
from mininet.log import setLogLevel

class NetworkTopo(Topo):
    def build(self):
        # Add hosts with default routes
        h1 = self.addHost('h1', ip='192.168.1.10/24', defaultRoute='via 192.168.1.1')
        h2 = self.addHost('h2', ip='192.168.1.20/24', defaultRoute='via 192.168.1.1')
        h3 = self.addHost('h3', ip='192.168.2.10/24', defaultRoute='via 192.168.2.1')
        h4 = self.addHost('h4', ip='192.168.2.20/24', defaultRoute='via 192.168.2.1')

        # Add switches
        s1 = self.addSwitch('s1')
        s2 = self.addSwitch('s2')

        # Add routers (as standard hosts)
        r1 = self.addHost('r1')
        r2 = self.addHost('r2')

        # Create links
        self.addLink(h1, s1)
        self.addLink(h2, s1)
        self.addLink(s1, r1)
        self.addLink(r1, r2)
        self.addLink(r2, s2)
        self.addLink(h3, s2)
        self.addLink(h4, s2)

def configure_network(net):
    # Retrieve router nodes
    r1, r2 = net.get('r1', 'r2')

    # Configure router interfaces
    r1.setIP('192.168.1.1/24', intf='r1-eth0')
    r1.setIP('10.0.0.1/30', intf='r1-eth1')
    r2.setIP('10.0.0.2/30', intf='r2-eth0')
    r2.setIP('192.168.2.1/24', intf='r2-eth1')

    # Enable IP forwarding
    r1.cmd('sysctl -w net.ipv4.ip_forward=1')
    r2.cmd('sysctl -w net.ipv4.ip_forward=1')

    # Add static routes
    r1.cmd('ip route add 192.168.2.0/24 via 10.0.0.2')
    r2.cmd('ip route add 192.168.1.0/24 via 10.0.0.1')

if __name__ == '__main__':
    setLogLevel('info')
    net = Mininet(topo=NetworkTopo())
    net.start()
    configure_network(net)
    CLI(net)
    net.stop()