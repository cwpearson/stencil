#pragma once
#define AC_FOR_USER_INT_PARAM_TYPES(FUNC)\
FUNC(AC_nx)\
FUNC(AC_ny)\
FUNC(AC_nz)\
FUNC(AC_mx)\
FUNC(AC_my)\
FUNC(AC_mz)\
FUNC(AC_nx_min)\
FUNC(AC_ny_min)\
FUNC(AC_nz_min)\
FUNC(AC_nx_max)\
FUNC(AC_ny_max)\
FUNC(AC_nz_max)\
FUNC(AC_mxy)\
FUNC(AC_nxy)\
FUNC(AC_nxyz)\
FUNC(AC_max_steps)\
FUNC(AC_save_steps)\
FUNC(AC_bin_steps)\
FUNC(AC_start_step)\
FUNC(AC_bc_type_top_x)\
FUNC(AC_bc_type_bot_x)\
FUNC(AC_bc_type_top_y)\
FUNC(AC_bc_type_bot_y)\
FUNC(AC_bc_type_top_z)\
FUNC(AC_bc_type_bot_z)

#define AC_FOR_USER_INT3_PARAM_TYPES(FUNC)\
FUNC(AC_global_grid_n)\
FUNC(AC_multigpu_offset)

#define AC_FOR_USER_REAL_PARAM_TYPES(FUNC)\
FUNC(AC_dsx)\
FUNC(AC_dsy)\
FUNC(AC_dsz)\
FUNC(AC_inv_dsx)\
FUNC(AC_inv_dsy)\
FUNC(AC_inv_dsz)\
FUNC(AC_dt)\
FUNC(AC_max_time)\
FUNC(AC_dsmin)\
FUNC(AC_xlen)\
FUNC(AC_ylen)\
FUNC(AC_zlen)\
FUNC(AC_xorig)\
FUNC(AC_yorig)\
FUNC(AC_zorig)\
FUNC(AC_unit_density)\
FUNC(AC_unit_velocity)\
FUNC(AC_unit_length)\
FUNC(AC_unit_magnetic)\
FUNC(AC_star_pos_x)\
FUNC(AC_star_pos_y)\
FUNC(AC_star_pos_z)\
FUNC(AC_M_star)\
FUNC(AC_sink_pos_x)\
FUNC(AC_sink_pos_y)\
FUNC(AC_sink_pos_z)\
FUNC(AC_M_sink)\
FUNC(AC_M_sink_init)\
FUNC(AC_M_sink_Msun)\
FUNC(AC_soft)\
FUNC(AC_accretion_range)\
FUNC(AC_switch_accretion)\
FUNC(AC_cdt)\
FUNC(AC_cdtv)\
FUNC(AC_cdts)\
FUNC(AC_nu_visc)\
FUNC(AC_cs_sound)\
FUNC(AC_eta)\
FUNC(AC_mu0)\
FUNC(AC_cp_sound)\
FUNC(AC_gamma)\
FUNC(AC_cv_sound)\
FUNC(AC_lnT0)\
FUNC(AC_lnrho0)\
FUNC(AC_zeta)\
FUNC(AC_trans)\
FUNC(AC_bin_save_t)\
FUNC(AC_ampl_lnrho)\
FUNC(AC_ampl_uu)\
FUNC(AC_angl_uu)\
FUNC(AC_lnrho_edge)\
FUNC(AC_lnrho_out)\
FUNC(AC_ampl_aa)\
FUNC(AC_init_k_wave)\
FUNC(AC_init_sigma_hel)\
FUNC(AC_forcing_magnitude)\
FUNC(AC_relhel)\
FUNC(AC_kmin)\
FUNC(AC_kmax)\
FUNC(AC_forcing_phase)\
FUNC(AC_k_forcex)\
FUNC(AC_k_forcey)\
FUNC(AC_k_forcez)\
FUNC(AC_kaver)\
FUNC(AC_ff_hel_rex)\
FUNC(AC_ff_hel_rey)\
FUNC(AC_ff_hel_rez)\
FUNC(AC_ff_hel_imx)\
FUNC(AC_ff_hel_imy)\
FUNC(AC_ff_hel_imz)\
FUNC(AC_G_const)\
FUNC(AC_GM_star)\
FUNC(AC_unit_mass)\
FUNC(AC_sq2GM_star)\
FUNC(AC_cs2_sound)

#define AC_FOR_USER_REAL3_PARAM_TYPES(FUNC)

#define AC_FOR_VTXBUF_HANDLES(FUNC)\
FUNC(VTXBUF_LNRHO)\
FUNC(VTXBUF_UUX)\
FUNC(VTXBUF_UUY)\
FUNC(VTXBUF_UUZ)\
FUNC(VTXBUF_AX)\
FUNC(VTXBUF_AY)\
FUNC(VTXBUF_AZ)\
FUNC(VTXBUF_ENTROPY)

#define AC_FOR_SCALARARRAY_HANDLES(FUNC)

#define STREAM_0 (0)
#define STREAM_1 (1)
#define STREAM_2 (2)
#define STREAM_3 (3)
#define STREAM_4 (4)
#define STREAM_5 (5)
#define STREAM_6 (6)
#define STREAM_7 (7)
#define STREAM_8 (8)
#define STREAM_9 (9)
#define STREAM_10 (10)
#define STREAM_11 (11)
#define STREAM_12 (12)
#define STREAM_13 (13)
#define STREAM_14 (14)
#define STREAM_15 (15)
#define STREAM_16 (16)
#define STREAM_17 (17)
#define STREAM_18 (18)
#define STREAM_19 (19)
#define STREAM_20 (20)
#define STREAM_21 (21)
#define STREAM_22 (22)
#define STREAM_23 (23)
#define STREAM_24 (24)
#define STREAM_25 (25)
#define STREAM_26 (26)
#define STREAM_27 (27)
#define STREAM_28 (28)
#define STREAM_29 (29)
#define STREAM_30 (30)
#define STREAM_31 (31)
#define NUM_STREAMS (32)
#define STREAM_DEFAULT (STREAM_0)
#define STREAM_ALL (NUM_STREAMS)
typedef int Stream;

#define AC_FOR_RTYPES(FUNC)\
FUNC(RTYPE_MAX)\
FUNC(RTYPE_MIN)\
FUNC(RTYPE_RMS)\
FUNC(RTYPE_RMS_EXP)\
FUNC(RTYPE_ALFVEN_MAX)\
FUNC(RTYPE_ALFVEN_MIN)\
FUNC(RTYPE_ALFVEN_RMS)\
FUNC(RTYPE_SUM)
