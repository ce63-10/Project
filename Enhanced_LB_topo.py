import pdb
from mininet.topo import Topo
from mininet.log import info

class MyTopo( Topo ):
    def __init__(self):
        "Create custom topo"
        # init Topology
        Topo.__init__(self)

        #Add hosts 
        info('*** Adding Hosts\n')
        Host = []
        number_host = 2
        bw_eth = 10
        for i in ['h'+str(i+1) for i in range(number_host)]:
            Host.append(self.addHost(i))

        #Add switchs
        info('*** Adding Switches\n')
        Switch = []
        number_switch = 8
        for i in ['s'+str(i+1) for i in range(number_switch)]:
            Switch.append(self.addSwitch(i))	

        #Add Links
        info('*** Creating Links (Host -- Switch)\n')
        self.addLink(Switch[0], Host[0], bw = bw_eth)
        self.addLink(Switch[3], Host[1], bw = bw_eth)

        info('*** Creating Links (Switch -- Switch)\n')
        self.addLink(Switch[0], Switch[1], bw = bw_eth)
        self.addLink(Switch[0], Switch[4], bw = bw_eth)
        self.addLink(Switch[0], Switch[7], bw = bw_eth)
        self.addLink(Switch[1], Switch[2], bw = bw_eth)
        self.addLink(Switch[2], Switch[3], bw = bw_eth)
        self.addLink(Switch[2], Switch[7], bw = bw_eth)
        self.addLink(Switch[3], Switch[6], bw = bw_eth)
        self.addLink(Switch[4], Switch[5], bw = bw_eth)
        self.addLink(Switch[5], Switch[6], bw = bw_eth)

topos = { 'mytopo': ( lambda: MyTopo() ) }
