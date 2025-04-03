from mininet.topo import Topo
from mininet.net import Mininet
from mininet.node import Controller, OVSSwitch
from mininet.cli import CLI
from mininet.log import setLogLevel

class ComplexTopology(Topo):
    def build(self):
        # Create switches (8 switches in a complex structure)
        s1 = self.addSwitch('s1', stp=True, prio=1)
        s2 = self.addSwitch('s2', stp=True, prio=2)
        s3 = self.addSwitch('s3', stp=True, prio=3)
        s4 = self.addSwitch('s4', stp=True, prio=4)
        s5 = self.addSwitch('s5', stp=True, prio=5)
        s6 = self.addSwitch('s6', stp=True, prio=6)
        s7 = self.addSwitch('s7', stp=True, prio=7)
        s8 = self.addSwitch('s8', stp=True, prio=8)

        # Create hosts (6 hosts)
        h1 = self.addHost('h1')
        h2 = self.addHost('h2')
        h3 = self.addHost('h3')
        h4 = self.addHost('h4')
        h5 = self.addHost('h5')
        h6 = self.addHost('h6')

        # Add links between switches (creating loops and redundant paths)
        self.addLink(s1, s2, bw=10, delay='5ms')
        self.addLink(s1, s3, bw=10, delay='5ms')
        self.addLink(s1, s4, bw=10, delay='5ms')
        self.addLink(s2, s3, bw=10, delay='5ms')  # Loop
        self.addLink(s2, s5, bw=10, delay='5ms')
        self.addLink(s3, s6, bw=10, delay='5ms')
        self.addLink(s4, s5, bw=10, delay='5ms')
        self.addLink(s4, s7, bw=10, delay='5ms')
        self.addLink(s5, s6, bw=10, delay='5ms')  # Loop
        self.addLink(s5, s8, bw=10, delay='5ms')
        self.addLink(s6, s8, bw=10, delay='5ms')
        self.addLink(s7, s8, bw=10, delay='5ms')  # Loop

        # Connect hosts to switches
        self.addLink(h1, s1, bw=100, delay='1ms')
        self.addLink(h2, s2, bw=100, delay='1ms')
        self.addLink(h3, s3, bw=100, delay='1ms')
        self.addLink(h4, s5, bw=100, delay='1ms')
        self.addLink(h5, s7, bw=100, delay='1ms')
        self.addLink(h6, s8, bw=100, delay='1ms')

def setup_network():
    topo = ComplexTopology()
    net = Mininet(topo=topo, controller=Controller, switch=OVSSwitch)
    net.start()
    
    # Wait for STP to converge (takes about 30-40 seconds)
    print("Waiting for STP to converge...")
    import time
    time.sleep(40)
    
    return net

if __name__ == '__main__':
    setLogLevel('info')
    net = setup_network()
    CLI(net)
    net.stop()