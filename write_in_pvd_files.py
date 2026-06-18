###################################################################################################
# MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW
###################################################################################################


def write_in_pvd_files(
    pvd_solution_file,
    pvd_swarm_file,
    istep,
    nstep,
    every_solution,
    every_swarm,
    geological_time,
):

    if istep == 0:
        pvd_solution_file.write('<?xml version="1.0"?> \n')
        pvd_solution_file.write('<VTKFile type="Collection" version="0.1" ByteOrder="LittleEndian"> \n')
        pvd_solution_file.write("  <Collection> \n")
        pvd_swarm_file.write('<?xml version="1.0"?> \n')
        pvd_swarm_file.write('<VTKFile type="Collection" version="0.1" ByteOrder="LittleEndian"> \n')
        pvd_swarm_file.write("  <Collection> \n")

    if istep % every_solution == 0 or istep == nstep - 1:
        pvd_solution_file.write(
            '    <DataSet timestep="%s" group="" part="0" file="solution_%04d.vtu"/>  \n' % (geological_time, istep)
        )
        pvd_solution_file.flush()

    if istep % every_swarm == 0 or istep == nstep - 1:
        pvd_swarm_file.write(
            '    <DataSet timestep="%s" group="" part="0" file="SWARM/swarm_%04d.vtu"/>  \n' % (geological_time, istep)
        )
        pvd_swarm_file.flush()

    return


###################################################################################################
