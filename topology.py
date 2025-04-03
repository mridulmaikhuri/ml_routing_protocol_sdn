from mininet.topo import Topo
from mininet.net import Mininet
from mininet.node import OVSController
from mininet.cli import CLI
from mininet.log import setLogLevel, info
from mininet.link import TCLink

class ComplexLoopTopo(Topo):
    """
    Complex topology with multiple loops:
    
    This topology consists of 10 switches (s1-s10) and 10 hosts (h1-h10).
    The switches are connected in multiple loops to create redundant paths.
    Each host is connected to one switch.
    
    Basic Structure:
    - Core loop: s1 -- s2 -- s3 -- s4 -- s1
    - Edge loops: s5 -- s6 -- s7 -- s5, s8 -- s9 -- s10 -- s8
    - Cross-connections: s1 -- s5, s2 -- s8, s3 -- s7, s4 -- s10
    """
    
    def build(self):
        # Add switches
        s1 = self.addSwitch('s1')
        s2 = self.addSwitch('s2')
        s3 = self.addSwitch('s3')
        s4 = self.addSwitch('s4')
        s5 = self.addSwitch('s5')
        s6 = self.addSwitch('s6')
        s7 = self.addSwitch('s7')
        s8 = self.addSwitch('s8')
        s9 = self.addSwitch('s9')
        s10 = self.addSwitch('s10')
        
        # Add hosts
        h1 = self.addHost('h1')
        h2 = self.addHost('h2')
        h3 = self.addHost('h3')
        h4 = self.addHost('h4')
        h5 = self.addHost('h5')
        h6 = self.addHost('h6')
        h7 = self.addHost('h7')
        h8 = self.addHost('h8')
        h9 = self.addHost('h9')
        h10 = self.addHost('h10')
        
        # Connect hosts to switches
        self.addLink(h1, s1)
        self.addLink(h2, s2)
        self.addLink(h3, s3)
        self.addLink(h4, s4)
        self.addLink(h5, s5)
        self.addLink(h6, s6)
        self.addLink(h7, s7)
        self.addLink(h8, s8)
        self.addLink(h9, s9)
        self.addLink(h10, s10)
        
        # Create core loop
        self.addLink(s1, s2)
        self.addLink(s2, s3)
        self.addLink(s3, s4)
        self.addLink(s4, s1)
        
        # Create edge loops
        self.addLink(s5, s6)
        self.addLink(s6, s7)
        self.addLink(s7, s5)
        
        self.addLink(s8, s9)
        self.addLink(s9, s10)
        self.addLink(s10, s8)
        
        # Create cross-connections
        self.addLink(s1, s5)
        self.addLink(s2, s8)
        self.addLink(s3, s7)
        self.addLink(s4, s10)
        
        # Add some links with different bandwidths and delays to test QoS
        self.addLink(s2, s9, cls=TCLink, bw=10, delay='5ms')
        self.addLink(s3, s6, cls=TCLink, bw=5, delay='10ms')
        self.addLink(s4, s7, cls=TCLink, bw=7, delay='15ms')

def createTopology():
    """
    Create and start the topology
    """
    topo = ComplexLoopTopo()
    net = Mininet(topo=topo, controller=OVSController)
    net.start()
    
    # Enable STP on all switches to handle loops
    for switch in net.switches:
        switch.cmd('ovs-vsctl set bridge ' + switch.name + ' stp_enable=true')
    
    info('*** Running CLI\n')
    CLI(net)
    
    info('*** Stopping network\n')
    net.stop()

if __name__ == '__main__':
    setLogLevel('info')
    createTopology()