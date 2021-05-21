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
        number_host = 6
        bw_eth = 10
        for i in ['h'+str(i+1) for i in range(number_host)]:
            Host.append(self.addHost(i))

        #Add switchs
        info('*** Adding Switches\n')
        Switch = []
        number_switch = 6
        for i in ['s'+str(i+1) for i in range(number_switch)]:
            Switch.append(self.addSwitch(i))	

        #Add Links
        info('*** Creating Links (Host -- Switch)\n')
        for i in range(number_switch):
            self.addLink(Switch[i], Host[i], bw = bw_eth)

        for i in range(number_switch):
            for j in range(i + 1, number_switch):
                self.addLink(Switch[i], Switch[j], bw = bw_eth)
        
topos = { 'mytopo': ( lambda: MyTopo() ) }
