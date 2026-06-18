###################################################################################################
# MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW
###################################################################################################

import numpy as np

###################################################################################################


def quadrature_setup(nq_per_dim, nel):

    match nq_per_dim:
        case 3:
            qcoords = [-np.sqrt(3.0 / 5.0), 0.0, np.sqrt(3.0 / 5.0)]
            qweights = [5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0]
        case 4:
            qc4a = np.sqrt(3.0 / 7.0 + 2.0 / 7.0 * np.sqrt(6.0 / 5.0))
            qc4b = np.sqrt(3.0 / 7.0 - 2.0 / 7.0 * np.sqrt(6.0 / 5.0))
            qw4a = (18 - np.sqrt(30.0)) / 36.0
            qw4b = (18 + np.sqrt(30.0)) / 36.0
            qcoords = [-qc4a, -qc4b, qc4b, qc4a]
            qweights = [qw4a, qw4b, qw4b, qw4a]
        case 5:
            qc5a = np.sqrt(5.0 + 2.0 * np.sqrt(10.0 / 7.0)) / 3.0
            qc5b = np.sqrt(5.0 - 2.0 * np.sqrt(10.0 / 7.0)) / 3.0
            qc5c = 0.0
            qw5a = (322.0 - 13.0 * np.sqrt(70.0)) / 900.0
            qw5b = (322.0 + 13.0 * np.sqrt(70.0)) / 900.0
            qw5c = 128.0 / 225.0
            qcoords = [-qc5a, -qc5b, qc5c, qc5b, qc5a]
            qweights = [qw5a, qw5b, qw5c, qw5b, qw5a]
        case _:
            exit("unknown nq_per_dim")

    nq_per_element = nq_per_dim**2
    nq = nq_per_element * nel

    return qcoords, qweights, nq_per_element, nq


###################################################################################################
