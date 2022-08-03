# -*- coding: utf-8 -*-
"""A list of user-defined signals

This file contains a list of user-defined signals.
Signals are either profiles or channels and can be defined for a list of different
machines.

"""

import numpy as np
import sys

from frnn_loader.primitives.signal import Signal
from frnn_loader.primitives.signal import signal_0d
from frnn_loader.primitives.profilesignal import ProfileSignal
from frnn_loader.primitives.channelsignal import ChannelSignal

from frnn_loader.backends.machine import MachineD3D, MachineNSTX, MachineJET


# List of all defined machines
all_machines = [MachineD3D, MachineJET]
# Number of channels to be used for all profiles
profile_num_channels = 128

# ZIPFIT comes from actual measurements
# jet and d3d:
# Define ProfileSignal objects for various signals
edens_profile_thomson = ProfileSignal(
    "Electron density profile Thomson Scattering",
    ["ELECTRONS/TS.BLESSED.CORE.DENSITY"],
    [MachineD3D],
    mapping_paths=[None],
    causal_shifts=[0],
    num_channels=profile_num_channels,
    data_avail_tolerances=[0.029],
)

qpsi_efitrt2 = ProfileSignal(
    "q profile efitrt2",
    ["EFITRT2/RESULTS.GEQDSK.QPSI"],
    [MachineD3D],
    mapping_paths=[None],
    causal_shifts=[0],
    num_channels=profile_num_channels,
    data_avail_tolerances=[0.029],
)

qpsi_efit01 = ProfileSignal(
    "q profile efit01",
    ["EFIT01/RESULTS.GEQDSK.QPSI"],
    [MachineD3D],
    mapping_paths=[None],
    causal_shifts=[0],
    num_channels=profile_num_channels,
    data_avail_tolerances=[0.029],
)

qpsi_efitrt1 = ProfileSignal(
    "q profile efitrt1",
    ["EFITRT1/RESULTS.GEQDSK.QPSI"],
    [MachineD3D],
    mapping_paths=[None],
    causal_shifts=[0],
    num_channels=profile_num_channels,
    data_avail_tolerances=[0.029],
)

etemp_profile_thomson = ProfileSignal(
    "Electron temperature profile Thomson Scattering",
    ["ELECTRONS/TS.BLESSED.CORE.TEMP"],
    [MachineD3D],
    mapping_paths=[None],
    causal_shifts=[0],
    num_channels=profile_num_channels,
    data_avail_tolerances=[0.029],
)

edens_profile = ProfileSignal(
    "Electron density profile",
    ["ppf/hrts/ne", "ZIPFIT01/PROFILES.EDENSFIT"],
    [MachineJET, MachineD3D],
    mapping_paths=["ppf/hrts/rho", None],
    causal_shifts=[0, 10],
    num_channels=profile_num_channels,
    data_avail_tolerances=[0.05, 0.02],
)

etemp_profile = ProfileSignal(
    "Electron temperature profile",
    ["ppf/hrts/te", "ZIPFIT01/PROFILES.ETEMPFIT"],
    [MachineJET, MachineD3D],
    mapping_paths=["ppf/hrts/rho", None],
    causal_shifts=[0, 10],
    num_channels=profile_num_channels,
    data_avail_tolerances=[0.05, 0.02],
)

pres_prime_profile = ProfileSignal(
    "Pressure gradient profile",
    ["ppf/hrts/te", "EFITRT1/RESULTS.GEQDSK.PRES"],
    [MachineJET, MachineD3D],
    mapping_paths=["ppf/hrts/rho", None],
    causal_shifts=[0, 0],
    num_channels=profile_num_channels,
    data_avail_tolerances=[0.05, 0.02],
)

mpi66m322d_spec_profile = ProfileSignal(
    "mpi66m322d_spectrogram",
    ["", "mpi66m322d_spec"],
    [MachineJET, MachineD3D],
    mapping_paths=["ppf/hrts/rho", None],
    causal_shifts=[0, 2.58],
    num_channels=profile_num_channels,
    data_avail_tolerances=[0.05, 0.02],
)

etemp_profilet = ProfileSignal(
    "Electron temperature profile tol",
    ["ppf/hrts/te", "ZIPFIT01/PROFILES.ETEMPFIT"],
    [MachineJET, MachineD3D],
    mapping_paths=["ppf/hrts/rho", None],
    causal_shifts=[0, 10],
    num_channels=profile_num_channels,
    data_avail_tolerances=[0.05, 0.029],
)

edens_profilet = ProfileSignal(
    "Electron density profile tol",
    ["ppf/hrts/ne", "ZIPFIT01/PROFILES.EDENSFIT"],
    [MachineJET, MachineD3D],
    mapping_paths=["ppf/hrts/rho", None],
    causal_shifts=[0, 10],
    num_channels=profile_num_channels,
    data_avail_tolerances=[0.05, 0.029],
)

itemp_profile = ProfileSignal(
    "Ion temperature profile",
    ["ZIPFIT01/PROFILES.ITEMPFIT"],
    [MachineD3D],
    causal_shifts=[10],
    mapping_range=(0, 1),
    num_channels=profile_num_channels,
    data_avail_tolerances=[0.02],
)

zdens_profile = ProfileSignal(
    "Impurity density profile",
    ["ZIPFIT01/PROFILES.ZDENSFIT"],
    [MachineD3D],
    causal_shifts=[10],
    num_channels=profile_num_channels,
    data_avail_tolerances=[0.02],
)

trot_profile = ProfileSignal(
    "Rotation profile",
    ["ZIPFIT01/PROFILES.TROTFIT"],
    [MachineD3D],
    causal_shifts=[10],
    num_channels=profile_num_channels,
    data_avail_tolerances=[0.02],
)

# note, thermal pressure doesn't include fast ions
pthm_profile = ProfileSignal(
    "Thermal pressure profile",
    ["ZIPFIT01/PROFILES.PTHMFIT"],
    [MachineD3D],
    causal_shifts=[10],
    num_channels=profile_num_channels,
    data_avail_tolerances=[0.02],
)

neut_profile = ProfileSignal(
    "Neutrals profile",
    ["ZIPFIT01/PROFILES.NEUTFIT"],
    [MachineD3D],
    causal_shifts=[10],
    num_channels=profile_num_channels,
    data_avail_tolerances=[0.02],
)

# compare the following profile to just q95 0D signal
q_profile = ProfileSignal(
    "Q profile",
    ["ZIPFIT01/PROFILES.BOOTSTRAP.QRHO"],
    [MachineD3D],
    causal_shifts=[10],
    num_channels=profile_num_channels,
    data_avail_tolerances=[0.02],
)

bootstrap_current_profile = ProfileSignal(
    "Rotation profile",
    ["ZIPFIT01/PROFILES.BOOTSTRAP.JBS_SAUTER"],
    [MachineD3D],
    causal_shifts=[10],
    num_channels=profile_num_channels,
    data_avail_tolerances=[0.02],
)

# EFIT is the solution to the inverse problem from external magnetic
# measurements

# pressure might be unphysical since it is not constrained by measurements,
# only the EFIT which does not know about density and temperature

q_psi_profile = ProfileSignal(
    "Q(psi) profile",
    ["EFIT01/RESULTS.GEQDSK.QPSI"],
    [MachineD3D],
    causal_shifts=[10],
    num_channels=profile_num_channels,
    data_avail_tolerances=[0.02],
)

# epress_profile_spatial = ProfileSignal(
#     "Electron pressure profile", ["ppf/hrts/pe/"], [MachineJET], causal_shifts=[25],
#     mapping_range=(2, 4), num_channels=profile_num_channels)

etemp_profile_spatial = ProfileSignal(
    "Electron temperature profile",
    ["ppf/hrts/te"],
    [MachineJET],
    causal_shifts=[0],
    mapping_range=(2, 4),
    num_channels=profile_num_channels,
    data_avail_tolerances=[0.05],
)

edens_profile_spatial = ProfileSignal(
    "Electron density profile",
    ["ppf/hrts/ne"],
    [MachineJET],
    causal_shifts=[0],
    mapping_range=(2, 4),
    num_channels=profile_num_channels,
    data_avail_tolerances=[0.05],
)

rho_profile_spatial = ProfileSignal(
    "Rho at spatial positions",
    ["ppf/hrts/rho"],
    [MachineJET],
    causal_shifts=[0],
    mapping_range=(2, 4),
    num_channels=profile_num_channels,
    data_avail_tolerances=[0.05],
)

etemp = Signal(
    "electron temperature",
    ["ppf/hrtx/te0"],
    [MachineJET],
    causal_shifts=[25],
    data_avail_tolerances=[0.05],
)

q95 = signal_0d(
    "q95 safety factor",
    ["ppf/efit/q95", "EFITRT1/RESULTS.AEQDSK.Q95"],
    [MachineJET, MachineD3D],
    causal_shifts=[0, 0],
    normalize=False,
    data_avail_tolerances=[0.03, 0.02],
)

qmin = Signal(
    "Minimum safety factor",
    ["EFITRT1/RESULTS.AEQDSK.QMIN"],
    [MachineD3D],
    causal_shifts=[0],
    normalize=False,
    data_avail_tolerances=[0.02],
)

q95_EFITRT1 = Signal(
    "q95 safety factor in real time",
    ["ppf/efit/q95", "EFITRT1/RESULTS.AEQDSK.Q95"],
    [MachineJET, MachineD3D],
    causal_shifts=[0, 0],
    normalize=False,
    data_avail_tolerances=[0.03, 0.029],
)

q95_efitrt2 = Signal(
    "q95 safety factor in real time efitrt2",
    ["ppf/efit/q95", "EFITRT2/RESULTS.AEQDSK.Q95"],
    [MachineJET, MachineD3D],
    causal_shifts=[0, 0],
    normalize=False,
    data_avail_tolerances=[0.03, 0.029],
)

vd = Signal(
    "vertical displacement change",
    ["/vpsdfz"],
    [MachineD3D],
    causal_shifts=[0],
    normalize=False,
    data_avail_tolerances=[0.029],
)

q95t = Signal(
    "q95 safety factor tol",
    ["ppf/efit/q95", "EFIT01/RESULTS.AEQDSK.Q95"],
    [MachineJET, MachineD3D],
    causal_shifts=[15, 10],
    normalize=False,
    data_avail_tolerances=[0.03, 0.029],
)

# "d3d/ipsip" was used before, ipspr15V seems to be available for a
# superset of shots.
ipsip = Signal(
    "plasma current ipsip",
    ["jpf/da/c2-ipla", "ipsip"],
    [MachineJET, MachineD3D],
    is_ip=True,
)

ip = Signal(
    "plasma current",
    ["jpf/da/c2-ipla", "ipspr15V"],
    [MachineJET, MachineD3D],
    is_ip=True,
)

ipori = Signal(
    "plasma current",
    ["jpf/da/c2-ipla", "ipspr15V"],
    [MachineJET, MachineD3D],
    is_ip=True,
)

ipt = Signal(
    "plasma current tol",
    ["jpf/da/c2-ipla", "ipsip"],
    [MachineJET, MachineD3D],
    is_ip=True,
    data_avail_tolerances=[0.029, 0.029],
)

iptarget = Signal("plasma current target", ["ipsiptargt"], [MachineD3D])

iptargett = Signal(
    "plasma current target tol",
    ["ipsiptargt"],
    [MachineD3D],
    data_avail_tolerances=[0.029],
)

iperr = Signal("plasma current error", ["ipeecoil"], [MachineD3D])

iperrt = Signal(
    "plasma current error tol",
    ["ipeecoil"],
    [MachineD3D],
    data_avail_tolerances=[0.029],
)

li = Signal(
    "internal inductance", ["jpf/gs/bl-li<s", "efsli"], [MachineJET, MachineD3D]
)

lit = Signal(
    "internal inductance tol",
    ["jpf/gs/bl-li<s", "efsli"],
    [MachineJET, MachineD3D],
    data_avail_tolerances=[0.029, 0.029],
)

lm = Signal(
    "Locked mode amplitude", ["jpf/da/c2-loca", "dusbradial"], [MachineJET, MachineD3D]
)

lmt = Signal(
    "Locked mode amplitude tol",
    ["jpf/da/c2-loca", "dusbradial"],
    [MachineJET, MachineD3D],
    data_avail_tolerances=[0.029, 0.029],
)

n1_rms = Signal(
    "n1 finite frequency signals",
    ["mhd/mirnov.n1rms"],
    [MachineD3D],
    data_avail_tolerances=[0.029],
    causal_shifts=10,
)

n2_rms_10 = Signal(
    "n2 finite frequency signals_10ms",
    ["mhd/mirnov.n2rms"],
    [MachineD3D],
    data_avail_tolerances=[0.029],
    causal_shifts=10,
)

n3_rms_10 = Signal(
    "n3 finite frequency signals_10ms",
    ["mhd/mirnov.n3rms"],
    [MachineD3D],
    data_avail_tolerances=[0.029],
    causal_shifts=10,
)

n1_rms_no_shift = Signal(
    "n1 finite frequency signals no shift",
    ["mhd/mirnov.n1rms"],
    [MachineD3D],
    data_avail_tolerances=[0.029],
)

dens = Signal(
    "Plasma density",
    ["jpf/df/g1r-lid:003", "dssdenest"],
    [MachineJET, MachineD3D],
    is_strictly_positive=True,
)

denst = Signal(
    "Plasma density tol",
    ["jpf/df/g1r-lid:003", "dssdenest"],
    [MachineJET, MachineD3D],
    is_strictly_positive=True,
    data_avail_tolerances=[0.029, 0.029],
)

energy = Signal(
    "stored energy", ["jpf/gs/bl-wmhd<s", "efswmhd"], [MachineJET, MachineD3D]
)

energyt = Signal(
    "stored energy tol",
    ["jpf/gs/bl-wmhd<s", "efswmhd"],
    [MachineJET, MachineD3D],
    data_avail_tolerances=[0.029, 0.029],
)

pin = Signal(
    "Input Power (beam for MachineD3D)",
    ["jpf/gs/bl-ptot<s", "bmspinj"],
    [MachineJET, MachineD3D],
)

pint = Signal(
    "Input Power (beam for MachineD3D) tol",
    ["jpf/gs/bl-ptot<s", "bmspinj"],
    [MachineJET, MachineD3D],
    data_avail_tolerances=[0.029, 0.029],
)

pradtot = Signal("Radiated Power", ["jpf/db/b5r-ptot>out"], [MachineJET])

pradtott = Signal(
    "Radiated Power tol",
    ["jpf/db/b5r-ptot>out"],
    [MachineJET],
    data_avail_tolerances=[0.029],
)

pradcore = ChannelSignal(
    "Radiated Power Core",
    ["ppf/bolo/kb5h/channel14", "/" + r"\bol_l15_p"],
    [MachineJET, MachineD3D],
)

pradedge = ChannelSignal(
    "Radiated Power Edge",
    ["ppf/bolo/kb5h/channel10", "/" + r"\bol_l03_p"],
    [MachineJET, MachineD3D],
)

pradcoret = ChannelSignal(
    "Radiated Power Core tol",
    ["ppf/bolo/kb5h/channel14", "/" + r"\bol_l15_p"],
    [MachineJET, MachineD3D],
    data_avail_tolerances=[0.029, 0.029],
)

pradedget = ChannelSignal(
    "Radiated Power Edge tol",
    ["ppf/bolo/kb5h/channel10", "/" + r"\bol_l03_p"],
    [MachineJET, MachineD3D],
    data_avail_tolerances=[0.029, 0.029],
)

pechin = Signal(
    "ECH input power, not always on", ["RF/ECH.TOTAL.ECHPWRC"], [MachineD3D]
)

pechint = Signal(
    "ECH input power, not always on tol",
    ["RF/ECH.TOTAL.ECHPWRC"],
    [MachineD3D],
    data_avail_tolerances=[0.029],
)

betan = Signal("Normalized Beta", ["efsbetan"], [MachineD3D])

betant = Signal(
    "Normalized Beta tol", ["efsbetan"], [MachineD3D], data_avail_tolerances=[0.029]
)

energydt = Signal("stored energy time derivative", ["jpf/gs/bl-fdwdt<s"], [MachineJET])

torquein = Signal("Input Beam Torque", ["/bmstinj"], [MachineD3D])

fs07 = signal_0d("filterscope fs07", ["fs07"], [MachineD3D])

neped = signal_0d("neped", ["prmtan_neped"], [MachineD3D])

newid = signal_0d("newid", ["prmtan_newid"], [MachineD3D])

peped = signal_0d("peped", ["prmtan_peped"], [MachineD3D])

teped = signal_0d("teped", ["prmtan_teped"], [MachineD3D])

tewid = signal_0d("tewid", ["prmtan_tewid"], [MachineD3D])

torqueint = Signal(
    "Input Beam Torque tol", ["bmstinj"], [MachineD3D], data_avail_tolerances=[0.029]
)

tmamp1 = Signal("Tearing Mode amplitude (rotating 2/1)", ["nssampn1l"], [MachineD3D])

tmamp2 = Signal("Tearing Mode amplitude (rotating 3/2)", ["nssampn2l"], [MachineD3D])

tmfreq1 = Signal("Tearing Mode frequency (rotating 2/1)", ["nssfrqn1l"], [MachineD3D])

tmfreq2 = Signal("Tearing Mode frequency (rotating 3/2)", ["nssfrqn2l"], [MachineD3D])

ipdirect = Signal("plasma current direction", ["iptdirect"], [MachineD3D])

ipdirectt = Signal(
    "plasma current direction tol",
    ["/ptdirect"],
    [MachineD3D],
    data_avail_tolerances=[0.029],
)

# for downloading, modify this to preprocess shots with only a subset of
# signals. This may produce more shots
# since only those shots that contain all_signals contained here are used.

#############################################################################################
#                                                                                           #
#                                Define groups of signals                                   #
#                                                                                           #
#############################################################################################
# Restricted subset to those signals that are present for most shots. The
# idea is to remove signals that cause many shots to be dropped from the
# dataset.
signal_group_gar18 = {
    "q95t": q95t,
    "lit": lit,
    "ipt": ipt,
    "betant": betant,
    "energyt": energyt,
    "lmt": lmt,
    "denst": denst,
    "pradcoret": pradcoret,
    "pradedget": pradedget,
    "pint": pint,
    "torqueint": torqueint,
    "ipdirectt": ipdirectt,
    "iptargett": iptargett,
    "iperrt": iperrt,
    "etemp_profilet": etemp_profilet,
    "edens_profilet": edens_profilet,
}

signal_group_real_time = {
    "q95_EFITRT1": q95_EFITRT1,
    "li": li,
    "ip": ip,
    "betan": betan,
    "energy": energy,
    "lm": lm,
    "dens": dens,
    "pradcore": pradcore,
    "pradedge": pradedge,
    "pradtot": pradtot,
    "pin": pin,
    "torquein": torquein,
    "energydt": energydt,
    "ipdirect": ipdirect,
    "iptarget": iptarget,
    "iperr": iperr,
    # 'tmamp1':tmamp1, 'tmamp2':tmamp2, 'tmfreq1':tmfreq1, 'tmfreq2':tmfreq2,
    # 'pechin':pechin,
    # 'rho_profile_spatial':rho_profile_spatial, 'etemp':etemp,
    "etemp_profile": etemp_profile,
    "edens_profile": edens_profile,
}
signal_group_real_time_0D = {
    "q95_EFITRT1": q95_EFITRT1,
    "li": li,
    "ip": ip,
    "betan": betan,
    "energy": energy,
    "lm": lm,
    "dens": dens,
    "pradcore": pradcore,
    "pradedge": pradedge,
    "pradtot": pradtot,
    "pin": pin,
    "torquein": torquein,
    "vd": vd,
    "energydt": energydt,
    "iperr": iperr,
    "ipdirect": ipdirect,
    "iptarget": iptarget
    # 'tmamp1':tmamp1, 'tmamp2':tmamp2, 'tmfreq1':tmfreq1, 'tmfreq2':tmfreq2,
    # 'pechin':pechin,
    # 'rho_profile_spatial':rho_profile_spatial, 'etemp':etemp,
}

signal_group_all = {
    "q95": q95,
    "li": li,
    "ip": ip,
    "betan": betan,
    "energy": energy,
    "lm": lm,
    "dens": dens,
    "pradcore": pradcore,
    "pradedge": pradedge,
    "pradtot": pradtot,
    "pin": pin,
    "torquein": torquein,
    "energydt": energydt,
    "ipdirect": ipdirect,
    "iptarget": iptarget,
    "iperr": iperr,
    # 'tmamp1':tmamp1, 'tmamp2':tmamp2, 'tmfreq1':tmfreq1, 'tmfreq2':tmfreq2,
    # 'pechin':pechin,
    # 'rho_profile_spatial':rho_profile_spatial, 'etemp':etemp,
    "etemp_profile": etemp_profile,
    "edens_profile": edens_profile,
    # 'itemp_profile':itemp_profile, 'zdens_profile':zdens_profile,
    # 'trot_profile':trot_profile, 'pthm_profile':pthm_profile,
    # 'neut_profile':neut_profile, 'q_profile':q_profile,
    # 'bootstrap_current_profile':bootstrap_current_profile,
    # 'q_psi_profile':q_psi_profile}
}
signal_group_ori = {
    "q95": q95,
    "li": li,
    "ipori": ipori,
    "betan": betan,
    "energy": energy,
    "lm": lm,
    "dens": dens,
    "pradcore": pradcore,
    "pradedge": pradedge,
    "pradtot": pradtot,
    "pin": pin,
    "torquein": torquein,
    "energydt": energydt,
    "ipdirect": ipdirect,
    "iptarget": iptarget,
    "iperr": iperr,
    "etemp_profile": etemp_profile,
    "edens_profile": edens_profile,
}

signal_group_n1rms = {
    "n1_rms_no_shift": n1_rms_no_shift,
    "n1_rms": n1_rms,
    "q95": q95,
    "li": li,
    "ip": ip,
    "betan": betan,
    "energy": energy,
    "lm": lm,
    "dens": dens,
    "pradcore": pradcore,
    "pradedge": pradedge,
    "pradtot": pradtot,
    "pin": pin,
    "torquein": torquein,
    "energydt": energydt,
    "ipdirect": ipdirect,
    "iptarget": iptarget,
    "iperr": iperr,
    "etemp_profile": etemp_profile,
    "edens_profile": edens_profile,
}

signal_group_n1rms_thomson = {
    "n1_rms_no_shift": n1_rms_no_shift,
    "n1_rms": n1_rms,
    "q95_EFITRT1": q95_EFITRT1,
    "li": li,
    "ip": ip,
    "betan": betan,
    "energy": energy,
    "lm": lm,
    "dens": dens,
    "pradcore": pradcore,
    "pradedge": pradedge,
    "pradtot": pradtot,
    "pin": pin,
    "torquein": torquein,
    "energydt": energydt,
    "ipdirect": ipdirect,
    "iptarget": iptarget,
    "iperr": iperr,
    # 'qpsi_efitrt2':qpsi_efitrt2,
    "etemp_profile_thomson": etemp_profile_thomson,
    "edens_profile_thomson": edens_profile_thomson,
}

signal_group_fs07 = {
    "n1_rms_no_shift": n1_rms_no_shift,
    "n1_rms": n1_rms,
    "q95": q95,
    "li": li,
    "ip": ip,
    "betan": betan,
    "energy": energy,
    "lm": lm,
    "dens": dens,
    "pradcore": pradcore,
    "fs07": fs07,
    "pradedge": pradedge,
    "pradtot": pradtot,
    "pin": pin,
    "torquein": torquein,
    "energydt": energydt,
    "ipdirect": ipdirect,
    "iptarget": iptarget,
    "iperr": iperr,
    "etemp_profile": etemp_profile,
    "edens_profile": edens_profile,
}

signal_group_ped_old_2021 = {
    "n1_rms_no_shift": n1_rms_no_shift,
    "n1_rms": n1_rms,
    "q95": q95,
    "li": li,
    "ip": ip,
    "betan": betan,
    "energy": energy,
    "lm": lm,
    "dens": dens,
    "pradcore": pradcore,
    "fs07": fs07,
    "pradedge": pradedge,
    "pradtot": pradtot,
    "pin": pin,
    "torquein": torquein,
    "neped": neped,
    "teped": teped,
    "newid": newid,
    "peped": peped,
    "tewid": tewid,
    "energydt": energydt,
    "ipdirect": ipdirect,
    "iptarget": iptarget,
    "iperr": iperr,
    "etemp_profile": etemp_profile,
    "edens_profile": edens_profile,
}

signal_group_ped = {
    "n1_rms": n1_rms,
    "n2_rms_10": n2_rms_10,
    "n3_rms_10": n3_rms_10,
    "q95": q95,
    "li": li,
    "ip": ip,
    "betan": betan,
    "energy": energy,
    "lm": lm,
    "dens": dens,
    "pradcore": pradcore,
    "fs07": fs07,
    "pradedge": pradedge,
    "pradtot": pradtot,
    "pin": pin,
    "torquein": torquein,
    "neped": neped,
    "teped": teped,
    "newid": newid,
    "peped": peped,
    "tewid": tewid,
    # 'energydt': energydt, 'ipdirect': ipdirect,
    "iptarget": iptarget,
    # 'iperr': iperr,
    "etemp_profile": etemp_profile,
    "edens_profile": edens_profile,
}


signal_group_ped_spec = {
    "qmin": qmin,
    "n1_rms": n1_rms,
    "n2_rms_10": n2_rms_10,
    "n3_rms_10": n3_rms_10,
    "q95": q95,
    "li": li,
    "ip": ip,
    "betan": betan,
    "energy": energy,
    "lm": lm,
    "dens": dens,
    "pradcore": pradcore,
    "fs07": fs07,
    "pradedge": pradedge,
    "pradtot": pradtot,
    "pin": pin,
    "torquein": torquein,
    "neped": neped,
    "teped": teped,
    "newid": newid,
    "peped": peped,
    "tewid": tewid,
    "iptarget": iptarget,
    "etemp_profile": etemp_profile,
    "edens_profile": edens_profile,
    "mpi66m322d_spec_profile": mpi66m322d_spec_profile,
    "qpsi_efitrt1": qpsi_efitrt1,
}


signal_group_n1rms_qmin = {
    "qmin": qmin,
    "n1_rms_no_shift": n1_rms_no_shift,
    "n1_rms": n1_rms,
    "q95": q95,
    "li": li,
    "ip": ip,
    "betan": betan,
    "energy": energy,
    "lm": lm,
    "dens": dens,
    "pradcore": pradcore,
    "pradedge": pradedge,
    "pradtot": pradtot,
    "pin": pin,
    "torquein": torquein,
    "energydt": energydt,
    "ipdirect": ipdirect,
    "iptarget": iptarget,
    "iperr": iperr,
    "etemp_profile": etemp_profile,
    "edens_profile": edens_profile,
}

signal_group_thomson = {
    "q95": q95,
    "li": li,
    "ip": ip,
    "betan": betan,
    "energy": energy,
    "lm": lm,
    "dens": dens,
    "pradcore": pradcore,
    "pradedge": pradedge,
    "pradtot": pradtot,
    "pin": pin,
    "torquein": torquein,
    "energydt": energydt,
    "ipdirect": ipdirect,
    "iptarget": iptarget,
    "iperr": iperr,
    "etemp_profile_thomson": etemp_profile_thomson,
    "edens_profile_thomson": edens_profile_thomson,
}


signal_group_restricted = signal_group_all

fully_defined_signals = {
    sig_name: sig
    for (sig_name, sig) in signal_group_restricted.items()
    if (all([m in sig.machines for m in all_machines]))
}

fully_defined_signals_0D = {
    sig_name: sig
    for (sig_name, sig) in signal_group_restricted.items()
    if (all([m in sig.machines for m in all_machines]))
}

fully_defined_signals_1D = {
    sig_name: sig
    for (sig_name, sig) in signal_group_restricted.items()
    if (all([m in sig.machines for m in all_machines]) and sig.num_channels > 1)
}

# All D3D signals
d3d_signals = {
    sig_name: sig
    for (sig_name, sig) in signal_group_restricted.items()
    if (MachineD3D in sig.machines)
}

d3d_signals_n1rms = {
    sig_name: sig
    for (sig_name, sig) in signal_group_n1rms.items()
    if (MachineD3D in sig.machines)
}

# All Signals with fs07
d3d_signals_fs07 = {
    sig_name: sig
    for (sig_name, sig) in signal_group_fs07.items()
    if (MachineD3D in sig.machines)
}

d3d_signals_ped = {
    sig_name: sig
    for (sig_name, sig) in signal_group_ped.items()
    if (MachineD3D in sig.machines)
}

d3d_signals_ped_spec = {
    sig_name: sig
    for (sig_name, sig) in signal_group_ped_spec.items()
    if (MachineD3D in sig.machines)
}

d3d_signals_gar18 = {
    sig_name: sig
    for (sig_name, sig) in signal_group_gar18.items()
    if (MachineD3D in sig.machines)
}

# All 1D signals on D3D
d3d_signals_0D = {
    sig_name: sig
    for (sig_name, sig) in signal_group_restricted.items()
    if (MachineD3D in sig.machines and sig.num_channels == 1)
}
# All 2D signals on D3D
d3d_signals_1D = {
    sig_name: sig
    for (sig_name, sig) in signal_group_restricted.items()
    if (MachineD3D in sig.machines and sig.num_channels > 1)
}

# All Signals on JET
jet_signals = {
    sig_name: sig
    for (sig_name, sig) in signal_group_restricted.items()
    if (MachineJET in sig.machines)
}

# All 0D signals on JET
jet_signals_0D = {
    sig_name: sig
    for (sig_name, sig) in signal_group_restricted.items()
    if (MachineJET in sig.machines and sig.num_channels == 1)
}
# All 1D signals on JET
jet_signals_1D = {
    sig_name: sig
    for (sig_name, sig) in signal_group_restricted.items()
    if (MachineJET in sig.machines and sig.num_channels > 1)
}

# End of file user_signals.py
