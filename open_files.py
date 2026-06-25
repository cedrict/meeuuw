###################################################################################################
# MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW
###################################################################################################


def open_files(vel_unit, time_unit):

    vrms_file = open("OUTPUT/vrms.ascii", "w")
    vrms_file.write("#time,vrms\n")

    pstats_file = open("OUTPUT/stats_pressure.ascii", "w")
    pstats_file.write("#istep,min p, max p\n")

    vstats_file = open("OUTPUT/stats_velocity.ascii", "w")
    vstats_file.write("#istep,min(u),max(u),min(v),max(v)\n")
    vstats_file.write("# " + vel_unit + "\n")

    srstats_file = open("OUTPUT/stats_strainrate.ascii", "w")
    srstats_file.write("#time min(e_n) max(e_n)\n")

    dt_file = open("OUTPUT/dt.ascii", "w")
    dt_file.write("#time dt1 dt2 dt\n")
    dt_file.write("#" + time_unit + "\n")

    ptcl_stats_file = open("OUTPUT/stats_particle.ascii", "w")
    ptcl_stats_file.write("#time min(nparticle_e) max(nparticle_e)\n")

    ptcl_active_file = open("OUTPUT/active_particles_number.ascii", "w")
    ptcl_active_file.write("#time active nparticle_needed nparticle_available )\n")

    timings_file = open("timings.ascii", "w")

    TM_file = open("OUTPUT/total_mass.ascii", "w")
    TM_file.write("# time total mass\n")

    EK_file = open("OUTPUT/kinetic_energy.ascii", "w")
    EK_file.write("# time kinetic_energy\n")

    TVD_file = open("OUTPUT/tvd.ascii", "w")
    TVD_file.write("# time total viscous dissipation\n")

    WAG_file = open("OUTPUT/wag.ascii", "w")
    WAG_file.write("#time WAG\n")

    T_avrg_file = open("OUTPUT/T_avrg.ascii", "w")
    T_avrg_file.write("#time <T>\n")

    eta_avrg_file = open("OUTPUT/eta_avrg.ascii", "w")
    eta_avrg_file.write("#time eta_avrg\n")

    delta_file = open("OUTPUT/delta_wag_tvd.ascii", "w")
    delta_file.write("#time delta max(abs(WAG),TVD)\n")

    pvd_solution_file = open("OUTPUT/solution.pvd", "w")
    pvd_swarm_file = open("OUTPUT/swarm.pvd", "w")

    etaq_file = open("OUTPUT/stats_eta_q.ascii", "w")
    etaq_file.write("#istep min(eta_q) max(eta_q)\n")

    etan_file = open("OUTPUT/stats_eta_n.ascii", "w")
    etan_file.write("#istep min(eta_n) max(eta_n)\n")

    etae_file = open("OUTPUT/stats_eta_e.ascii", "w")
    etae_file.write("#time min(eta_e) max(eta_e)\n")

    corner_q_file = open("OUTPUT/corner_heat_flux.ascii", "w")
    corner_q_file.write("# time qx0 qz0 qx1 qz1 qx2 qz2 qx3 qz3\n")

    Tstats_file = open("OUTPUT/stats_temperature.ascii", "w")

    Nu_file = open("OUTPUT/Nu.ascii", "w")
    Nu_file.write("#time Nu\n")

    avrg_T_bot_file = open("OUTPUT/bottom/avrg_T_bot.ascii", "w")
    avrg_T_top_file = open("OUTPUT/top/avrg_T_top.ascii", "w")

    avrg_dTdz_bot_file = open("OUTPUT/bottom/avrg_dTdz_bot.ascii", "w")
    avrg_dTdz_top_file = open("OUTPUT/top/avrg_dTdz_top.ascii", "w")

    conv_file = open("OUTPUT/conv.ascii", "w")


    return (
        vrms_file,
        pstats_file,
        vstats_file,
        srstats_file,
        dt_file,
        ptcl_stats_file,
        ptcl_active_file,
        timings_file,
        TM_file,
        EK_file,
        TVD_file,
        WAG_file,
        T_avrg_file,
        eta_avrg_file,
        delta_file,
        pvd_solution_file,
        pvd_swarm_file,
        etaq_file,
        etan_file,
        etae_file,
        corner_q_file,
        Tstats_file,
        Nu_file,
        avrg_T_bot_file,
        avrg_T_top_file,
        avrg_dTdz_bot_file,
        avrg_dTdz_top_file,
        conv_file
    )


###################################################################################################
