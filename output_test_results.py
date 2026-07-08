###################################################################################################
# MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW
###################################################################################################

import numpy as np

###################################################################################################


def output_test_results(m_V,x_V,z_V,Nfem_V,
                        m_P,x_P,z_P,Nfem_P,
                        m_T,x_T,z_T,Nfem_T,
                        nelx,nelz,vrms,T_avrg,output_folder):

    """
    bla
    """

    test_file = open(output_folder+"/output_test.ascii", "w")
    test_file.write("%d %d \n" % (nelx,nelz))
    test_file.write("%d %d %.5e %.5e \n" % (m_V,Nfem_V,np.min(x_V),np.max(x_V)))
    test_file.write("%d %d %.5e %.5e \n" % (m_P,Nfem_P,np.min(x_P),np.max(x_P)))
    test_file.write("%d %d %.5e %.5e \n" % (m_T,Nfem_T,np.min(x_T),np.max(x_T)))
    test_file.write("%.5e %.5e \n" % (vrms,T_avrg))
    #test_file.write("%.5e %.5e \n" % (np.min(area),np.max(area)))
    test_file.close()


###################################################################################################
