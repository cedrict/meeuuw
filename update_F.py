###################################################################################################
# MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW
###################################################################################################

import numpy as np
#from experiment4 import T_solidus,T_liquidus,fff

###############################################################################
# ---------------------> T
#  \          \
#   \          \
#    \          \
#   solidus    liquidus

def update_F(nparticle,swarm_p,swarm_T,swarm_F):
    """
    Args:
     nparticle: nb of particles
     swarm_p: array of pressure values for all particles
     swarm_T: array of temperature values for all particles
     swarm_F: current F values for all particles 
    Returns:
     swarm_F: updated F values for all particles 
    """

    swarm_sst=np.zeros(nparticle,dtype=np.float32)

    for ip in range(0,nparticle):

        Tsol=T_solidus(swarm_p[ip])
        Tliq=T_liquidus(swarm_p[ip])

        if swarm_T[ip]<=Tsol:
           super_solidus_temperature=0
        elif swarm_T[ip]<=Tliq:
           super_solidus_temperature=(swarm_T[ip]-Tsol)/(Tliq-Tsol) # theta
           swarm_F[ip]=max(fff(super_solidus_temperature),swarm_F[ip])
        else:
           super_solidus_temperature=1
        #end if
        swarm_sst[ip]=super_solidus_temperature
    #end for

    return swarm_F,swarm_sst

###############################################################################
