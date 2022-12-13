# -*- coding: utf-8 -*-
"""Propellant Storage System (VLM, DelfiPQ) Main Design Script.

Copyright (C) 2022  Lukas Welzel

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

@Author: Lukas Welzel
@Date: 07.12.2022
@Credit: Lukas Welzel/TU Delft

Todo:
    * Command line interface
    * Improve SMBO algo integration

"""

import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, minimize, shgo
from materials_library import *
from warnings import warn
from itertools import product
from tqdm import tqdm
from copy import deepcopy
from mipego import ContinuousSpace, OrdinalSpace, RandomForest, AnnealingBO

USE_GPU = True
if USE_GPU:
    bd = cp
else:
    bd = np

# <editor-fold desc="INPUTS: REQUIREMENTS, DESIGN OPTIONS, CONSTRAINTS">
### REQUIREMENTS
REQ_m_total_max = 32.0 * 1e-3  # g -> kg
REQ_V_total_max = 10.584 * 1e-5  # cm3 -> m3
REQ_MEOP = 5.5 * 1e5  # bar -> Pa
REQ_P_min = 1.0 * 1e5  # bar -> Pa
REQ_P_max = 5.0 * 1e5  # bar -> Pa
REQ_SF_yield = 1.6
REQ_SF_ult = 2.0

### DESIGN PARAMETERS
# materials
material_objects = np.array([Ti6Al4V, A7075T6, SST301FH, SST304])
material_properties_tensor = bd.array([
    # E Pa,         rho kg/m3,      sigma_yield Pa,         sigma_ult Pa
    [Ti6Al4V.E, Ti6Al4V.rho, Ti6Al4V.sigma_yield, Ti6Al4V.sigma_ult],
    [A7075T6.E, A7075T6.rho, A7075T6.sigma_yield, A7075T6.sigma_ult],
    [SST301FH.E, SST301FH.rho, SST301FH.sigma_yield, SST301FH.sigma_ult],
    [SST304.E, SST304.rho, SST304.sigma_yield, SST304.sigma_ult]
])
material_idxs = [0, 4]

# margins
m_pss_dry_margin = 0.15
V_tank_interior_margin = 0.15
V_tank_exterior_margin = 0.05

# production/manufacturing limits
# SM: sheet metal (min, max)
SM_feature_width = bd.array([1., 10.]) * 1e-3  # mm -> m
SM_thickness = bd.array([0.2, 6.]) * 1e-3  # mm -> m

# cops
leg_length_bounds = bd.array([2.5, 70.]) * 1e-3  # mm -> m
deflection_0 = - bd.array([20., 0.]) * 1e-3  # mm -> m
deflection_1 = bd.array([0., 25.]) * 1e-3  # mm -> m

# tank
tank_wall_thickness = bd.array([0.25, 10.0]) * 1e-3  # mm -> m
tank_cap_thickness = bd.array([0.25, 10.0]) * 1e-3  # mm -> m
tank_a_bounds = bd.array([7.5, 75.0]) * 1e-3  # mm -> m

# envelope
aspect_ratio = bd.array([1e-2, 1e2])  # r_equiv/h (not a strict requirement, only to restrict solution space)

# COPS
N_stages = bd.array([1, 50])
N_stages_cost = bd.array([1, 10])
N_legs = 3
# pre_deformation_limits = bd.array([0.01, 0.85])  # plastic deformation (pre-bending) as ratio of positive displacement

# cost factor: fraction of objective score (to be minimized) that can be increased by cost violations
cost_factor = 0.05

# constraint violation factor: fraction of objective score (to be minimized) that can be increased by cost violations
constraint_violation_factor = 1.

### CONSTANTS
K_Theta = 2.65
gamma = 0.85
propellant_density = 977.0  # kg/m3 at 4 DEG C

### DelfiPQ data
satellite_mass = 0.6  # kg
satellite_exit_velocity = 1000  # m/s
# </editor-fold>

# <editor-fold desc="PSS DATA ARRAY">
def make_pss_array(tank_material=0,
                   cops_material=0.,
                   n_stages=1.,
                   leg_width=5.0 * 1e-3,
                   leg_thickness=1.0 * 1e-3,
                   leg_length=15.0 * 1e-3,
                   displacement_0=-1.0 * 1e-3,
                   displacement_1=1.0 * 1e-3,
                   ):
    # TODO: MIPEGO issue: OrdinalSpace can output numbers outside given range (low likelihood)
    tank_material = np.maximum(np.minimum(np.around(tank_material), len(materials_list) - 1), 0.)
    cops_material = np.maximum(np.minimum(np.around(cops_material), len(materials_list) - 1), 0.)
    return bd.array([
                # GENERAL
                tank_material,  # tank material    0
                cops_material,  # cops material    1
                n_stages,  # .    N_stages         2
                3.,  # .          N_legs           3
                # COPS
                leg_width,  # .         leg width        4
                leg_thickness,  # .     leg thickness    5
                leg_length,  # .        leg length       6
                displacement_0,  # .    displacement_0   7
                displacement_1,  # .    displacement_1   8
                0.,  # E                 9
                0.,  # rho               10
                0.,  # sigma_yield       11
                0.,  # sigma_ult         12
                # TANK
                0.,  # tank a                       13
                0.,  # tank height                  14  # TODO: currently functionally evaluated locally, make backref
                0.,  # tank wall thickness          15
                0.,  # tank cap thickness           16
                0.,  # E                            17
                0.,  # rho                          18
                0.,  # sigma_yield                  19
                0.,  # sigma_ult                    20
                # INTERMEDIATE RESULTS
                0.,  # cops_material_sigma_critical 21
                0.,  # tank_material_sigma_critical 22
                # CONSTRAINTS
                0.,  # cops_max_force_max_displacement_constraint   23
                0.,  # cops_min_force_max_displacement_constraint   24
                0.,  # cops_max_force_zero_displacement_constraint  25
                0.,  # cops_min_force_zero_displacement_constraint  26
                0.,  # cops_max_sigma_max_displacement_constraint   27
                0.,  # cops_max_sigma_zero_displacement_constraint  28
                0.,  # cops_max_sigma_min_displacement_constraint   29
                0.,  # max_mass_constraint                          30
                0.,  # max_volume_constraint                        31
                # COST
                0.,  # COPS_width           32
                0.,  # COPS_thickness       33
                0.,  # COPS_length          34
                0.,  # N_stages             35
                # OUTPUT
                0.,  # propellant mass      36
                0.,  # tank mass            37
                0.,  # cops mass            38
                0.,  # PSS mass             39
                0.,  # PSS volume           40
            ])

# </editor-fold>

# <editor-fold desc="RESPONSE RELATIONS">
def get_cops_force_displacement(x, displacement_1):
    I = x[4] * x[5] ** 3 / 12
    Theta = x[13] * bd.sin(displacement_1 / (2 * gamma * x[6]))
    Theta_0 = x[13] * bd.sin(x[7] / (2 * gamma * x[6]))
    F = (9. / 6.) * (
            (12 * K_Theta * x[9] * I * (Theta - Theta_0))
            / (x[6] ** 2 * bd.cos(Theta))
    )
    return F


def get_cops_stress_displacement(x, displacement_1):
    c = x[5] / 2
    Theta = x[13] * bd.sin(displacement_1 / (2 * gamma * x[6]))
    max_stress = (2 * K_Theta * x[9] * c * (1 - gamma * (1 - bd.cos(Theta))) * Theta) / (
            x[6] * bd.cos(Theta))
    return max_stress


# </editor-fold>

# <editor-fold desc="DESIGN RELATIONS">
@cp.fuse()
def get_tank_cap_area(x):
    return 3.0 * bd.sqrt(3.) / 2 * bd.square(x[13])


@cp.fuse()
def get_pressure_force(x, P):
    return get_tank_cap_area(x) * P


@cp.fuse()
def get_tank_interior_height_from_cops(x):
    return x[2] * (x[8] + x[5])


@cp.fuse()
def get_tank_interior_volume(x):
    tank_interior_height = get_tank_interior_height_from_cops(x)
    tank_interior_area = get_tank_cap_area(x)
    return tank_interior_height * tank_interior_area


@cp.fuse()
def get_COPS_volume(x):
    return x[4] * x[5] * x[6] * 2 * 3

@cp.fuse()
def get_total_COPS_volume(x):
    return get_COPS_volume(x) * x[2]


@cp.fuse()
def get_COPS_mass(x):
    return get_COPS_volume(x) * x[10]

@cp.fuse()
def get_total_COPS_mass(x):
    return get_COPS_mass(x) * x[2]

@cp.fuse()
def get_propellant_volume(x):
    V_tank_interior = get_tank_interior_volume(x)
    V_COPS_total = get_total_COPS_volume(x)
    return (V_tank_interior - V_COPS_total) * (1. - V_tank_interior_margin)

@cp.fuse()
def get_propellant_mass(x):
    return get_propellant_volume(x) * propellant_density


def get_hex_tank_wall_thickness(x):
    tank_r_inner = x[13] / (2. * bd.tan(bd.pi / 6.))
    alpha = bd.pi / 6.
    t = (bd.sqrt(9 * REQ_MEOP ** 2 * tank_r_inner ** 2
                 + 12 * REQ_MEOP * tank_r_inner ** 2 * x[22] * bd.tan(alpha) ** 2)
         + 3 * REQ_MEOP * tank_r_inner) / (6 * x[22])
    return bd.clip(t, a_min=tank_wall_thickness[0], a_max=None)


def get_hex_tank_cap_thickness(x):
    tank_r_outer = x[13] / (2 * bd.sin(bd.pi / 6.))

    # min thickness at edges:
    t = bd.sqrt((3 * REQ_MEOP * tank_r_outer ** 2) / (4 * x[22]))
    return bd.clip(t, a_min=tank_wall_thickness[0], a_max=None)

def get_hex_tank_cap_deflection(x):
    tank_r_outer = x[13] / (2 * bd.sin(bd.pi / 6.))
    tank_cap_thickness = get_hex_tank_cap_thickness(x)
    deflection_center = 0.171 * REQ_P_max * tank_r_outer ** 4 / (x[17] * tank_cap_thickness ** 3)  # approx
    return deflection_center


def get_tank_wall_volume(x):
    tank_wall_thickness = get_hex_tank_wall_thickness(x)
    V_walls = 6 * x[13] * tank_wall_thickness * get_tank_interior_height_from_cops(x)
    return V_walls


def get_tank_caps_volume(x):
    cap_thickness = get_hex_tank_cap_thickness(x)
    V_caps = 2 * 3 * x[13] ** 2 / (2 * bd.tan(bd.pi / 6.)) * cap_thickness
    return V_caps


def get_tank_wall_mass(x):
    return get_tank_wall_volume(x) * x[18]


def get_tank_caps_mass(x):
    return get_tank_caps_volume(x) * x[18]


def get_tank_mass(x):
    return get_tank_wall_mass(x) + get_tank_caps_mass(x)


def get_dry_pss_mass(x):
    return (get_tank_mass(x) + get_total_COPS_mass(x)) * (1. + m_pss_dry_margin)


def get_wet_pss_mass(x):
    return get_dry_pss_mass(x) + get_propellant_mass(x)

def get_total_pss_volume(x):
    return get_tank_interior_volume(x) * (1. + V_tank_exterior_margin)


def ReLU_cupy(x):
    return x * (x > 0)

# </editor-fold>

# <editor-fold desc="CONSTRAINTS">
@cp.fuse()
def cops_max_force_max_displacement_constraint(x):
    """
    Actual force at full displacement must be smaller than the maximum force to comply with the maximum pressure
     requirement.
     :returns
        x > 0 if constraint satisfied
        x < 0 if constraint violated
    """

    force_actual = get_cops_force_displacement(x, x[8])
    force_required = get_pressure_force(x, REQ_P_max)
    return (force_required - force_actual) / force_required

@cp.fuse()
def cops_min_force_max_displacement_constraint(x):
    """
    Actual force at full displacement must be larger than the minimum force to comply with the minimum pressure
     requirement.
     :returns
        x > 0 if constraint satisfied
        x < 0 if constraint violated
    """
    force_actual = get_cops_force_displacement(x, x[8])
    force_required = get_pressure_force(x, REQ_P_min)
    return (force_actual - force_required) / force_actual

@cp.fuse()
def cops_max_force_zero_displacement_constraint(x):
    """
    Actual force at zero displacement (COPS is a plane) must be smaller than the maximum force to comply with the
     maximum pressure requirement.
     :returns
        x > 0 if constraint satisfied
        x < 0 if constraint violated
    """
    force_actual = get_cops_force_displacement(x, 0.)
    force_required = get_pressure_force(x, REQ_P_max)

    return (force_required - force_actual) / force_required

@cp.fuse()
def cops_min_force_zero_displacement_constraint(x):
    """
    Actual force at zero displacement (COPS is a plane) must be larger than the minimum force to comply with the
     minimum pressure requirement.
     :returns
        x > 0 if constraint satisfied
        x < 0 if constraint violated
    """
    force_actual = get_cops_force_displacement(x, 0.)
    force_required = get_pressure_force(x, REQ_P_min)

    return (force_actual - force_required) / force_actual

@cp.fuse()
def cops_max_sigma_max_displacement_constraint(x):
    """
    Actual stress at maximum displacement must be less than the maximum allowable stress.
     :returns
        x > 0 if constraint satisfied
        x < 0 if constraint violated
    """
    max_stress_actual = get_cops_stress_displacement(x, x[8])
    return (x[21] - max_stress_actual) / x[21]

@cp.fuse()
def cops_max_sigma_zero_displacement_constraint(x):
    """
    Actual stress at zero displacement (COPS is a plane) must be less than the maximum allowable stress.
     :returns
        x > 0 if constraint satisfied
        x < 0 if constraint violated
    """
    max_stress_actual = get_cops_stress_displacement(x, 0.)
    return (x[21] - max_stress_actual) / x[21]

@cp.fuse()
def cops_max_sigma_min_displacement_constraint(x):
    """
    Actual stress at zero displacement (COPS is a plane) must be less than the maximum allowable stress.
     :returns
        x > 0 if constraint satisfied
        x < 0 if constraint violated
    """
    max_stress_actual = get_cops_stress_displacement(x, - x[7])
    return (x[20] - max_stress_actual) / x[20]

@cp.fuse()
def constraint_function(theta):
    x = make_pss_array(*theta)
    # FORCE: max displacement constraints
    x[23] = cops_max_force_max_displacement_constraint(x)
    x[24] = cops_min_force_max_displacement_constraint(x)

    # FORCE: zero displacement constraints
    x[25] = cops_max_force_zero_displacement_constraint(x)
    x[26] = cops_min_force_zero_displacement_constraint(x)

    # Stress: constraints
    x[27] = cops_max_sigma_max_displacement_constraint(x)  # max displacement
    x[28] = cops_max_sigma_zero_displacement_constraint(x)  # zero displacement
    x[29] = cops_max_sigma_min_displacement_constraint(x)  # min displacement

    # REQUIREMENTS
    x[30] = REQ_m_total_max - (x[39] + x[36])   # max mass
    x[31] = REQ_V_total_max - x[40]  # max volume

    return x[23:32]

# </editor-fold>

# <editor-fold desc="OBJECTIVES">
@cp.fuse()
def objective_mass(pss_mass, propellant_mass):
    """
    Objective to minimize wet system mass.
    """
    return pss_mass + propellant_mass

@cp.fuse()
def objective_delta_V(pss_mass, propellant_mass):
    """
    Objective to maximize delta V implemented as minimizing negative delta V.
    """
    return - satellite_exit_velocity * bd.log((satellite_mass + pss_mass + propellant_mass)
                                              / (satellite_mass + pss_mass))


# </editor-fold>

# <editor-fold desc="DESIGN STEPS">
def set_material_properties(x):
    # stamp down the correct material properties into the x array design spot
    x[9:13] = material_properties_tensor[int(x[1]), :]
    x[17:21] = material_properties_tensor[int(x[1]), :]

    # select the critical tensile strength based on material properties and safety factors
    x[21] = bd.minimum(x[11] * REQ_SF_yield, x[12] * REQ_SF_ult)
    x[22] = bd.minimum(x[19] * REQ_SF_yield, x[20] * REQ_SF_ult)

    return x


def evaluate_constraints(x):
    # FORCE: max displacement constraints
    x[23] = cops_max_force_max_displacement_constraint(x)
    x[24] = cops_min_force_max_displacement_constraint(x)

    # FORCE: zero displacement constraints
    x[25] = cops_max_force_zero_displacement_constraint(x)
    x[26] = cops_min_force_zero_displacement_constraint(x)

    # Stress: constraints
    x[27] = cops_max_sigma_max_displacement_constraint(x)  # max displacement
    x[28] = cops_max_sigma_zero_displacement_constraint(x)  # zero displacement
    x[29] = cops_max_sigma_min_displacement_constraint(x)  # min displacement

    # REQUIREMENTS
    x[30] = REQ_m_total_max - (x[39] + x[36])   # max mass
    x[31] = REQ_V_total_max - x[40]  # max volume

    # rescale constraint costs to [0, 1)
    x[23:32] = ReLU_cupy(- x[23:32])  # positive values are compliant, change sign and apply ReLU
    x[23:32] = bd.tanh(x[23:32])  # rescale [0, +inf) -> [0, 1) using horizontal tangent

    return x


def validate_constraints(x):
    return cp.greater(bd.sum(x[23:32]), 0.) * (1.0 + bd.sum(x[23:32]))  # return 1 + sum of violations

def get_cost(para, lower=1e-9, upper=1.):
    """
    Compute cost [1., +inf) using an upper and lower bound.
    """
    return np.clip(para / upper, a_min=1., a_max=None) / np.clip(para / lower, a_min=None, a_max=1.)

def set_costs(x):
    x[32] = get_cost(x[4], SM_feature_width[0], SM_feature_width[1])  # COPS_width
    x[33] = get_cost(x[5], SM_thickness[0], SM_thickness[1])  # COPS_thickness
    x[34] = get_cost(x[6], leg_length_bounds[0], leg_length_bounds[1])  # COPS_length
    x[35] = get_cost(x[2], N_stages_cost[0], N_stages_cost[1])  # N_stages

    # rescale costs to [0, 1)
    x[32:36] = bd.tanh(x[32:36] - 1.)  # rescale [1., +inf) -> [0, 1) using horizontal tangent
    return x

def get_total_cost(x):
    return cp.greater(bd.sum(x[32:36]), 0.) * (bd.sum(x[32:36]))  # return sum of costs

def get_intermediate_design(x):
    x[13] = x[6] / 1.2  # assume that the COPS leg is a bit longer than a hexagon segment
    x[14] = get_tank_interior_height_from_cops(x)
    x[15] = get_hex_tank_wall_thickness(x)
    x[16] = get_hex_tank_cap_thickness(x)

    x[36] = get_propellant_mass(x)
    x[37] = get_tank_mass(x)
    x[38] = get_COPS_mass(x)
    x[39] = get_dry_pss_mass(x)

    x[40] = get_total_pss_volume(x)

    return x


def evaluate_design(x):
    # prepare for design step by setting design parameters
    x = set_material_properties(x)

    # compute configuration parameters
    x = get_intermediate_design(x)

    # validate constraints
    x = evaluate_constraints(x)
    constraint_violation = validate_constraints(x)  # if all satisfied 0, else 1 + sum of violations, each [0, 1)

    # compute costs
    x = set_costs(x)
    cost = get_total_cost(x)

    # objective score
    delta_V = objective_delta_V(x[39], x[36])

    # adjusted score (objective + penalties)
    score = delta_V * (1. - (cost * cost_factor + constraint_violation * constraint_violation_factor))

    return score, x, delta_V, cost, constraint_violation


def design_from_para(theta):
    # theta =
    # (tank_material,cops_material, n_stages, leg_width, leg_thickness, leg_length, displacement_0, displacement_1,)
    x = make_pss_array(*theta)
    return evaluate_design(x)  # score, x, delta_V, cost, constraint_violation

def pss_trial(theta):
    # theta =
    # (tank_material,cops_material, n_stages, leg_width, leg_thickness, leg_length, displacement_0, displacement_1,)
    x = make_pss_array(*theta)
    score, __, __, __, __ = evaluate_design(x)
    return score.get()

def mem_local_opt_pss_trial(theta):
    # theta =
    # (tank_material,cops_material, n_stages, leg_width, leg_thickness, leg_length, displacement_0, displacement_1,)
    x = make_pss_array(*theta)
    score, __, __, __, __ = evaluate_design(x)
    local_opt_trials.append(np.append(theta, score.get()))
    return score.get()
# </editor-fold>

# <editor-fold desc="OPTIMIZATION">
def optimize_pss_mipego(adaptive_fn_eval=50, warm_up_fn_eval=500, n_cores=16, n_per_iter=10, f_name="MIP_EGO_results.csv"):
    tank_material = OrdinalSpace(material_idxs, var_name='tank_material')
    cops_material = OrdinalSpace(material_idxs, var_name='cops_material')
    n_stages = OrdinalSpace(N_stages.tolist(), var_name='n_stages')
    leg_width = ContinuousSpace(SM_feature_width.tolist(), var_name='leg_width')
    leg_thickness = ContinuousSpace(SM_thickness.tolist(), var_name='leg_thickness')
    leg_length = ContinuousSpace(leg_length_bounds.tolist(), var_name='leg_length')
    displacement_0 = ContinuousSpace(deflection_0.tolist(), var_name='displacement_0')
    displacement_1 = ContinuousSpace(deflection_1.tolist(), var_name='displacement_1')

    search_space = tank_material + cops_material + n_stages + leg_width + leg_thickness + leg_length + displacement_0 + displacement_1

    model = RandomForest(levels=search_space.levels)
    opt = AnnealingBO(
        search_space=search_space,
        obj_fun=pss_trial,
        model=model,
        max_FEs=adaptive_fn_eval+warm_up_fn_eval,
        DoE_size=warm_up_fn_eval,
        # ineq_fun=constraint_function,  # doesnt work, split into jobs? Gets (theta0, theta1, ..., theta_n_jobs)?
        eval_type='list',
        acquisition_fun='MGFI',
        acquisition_par={'t': 2},
        n_job=n_cores,       # number of processes
        n_point=n_per_iter,     # number of the candidate solution proposed in each iteration
        verbose=True,
        data_file=f_name,
    )

    xopt, fopt, stop_dict = opt.run()

    print("<=== DONE ===>")

    return(xopt, fopt, stop_dict)

def optimize_from_guess(guess_csv_path="", local_iter=10,
                        max_score=-20.0, keep_track=True):
    import pandas as pd
    from visualize_results import theta2design
    from tqdm import tqdm

    columns = [
        "process_id",
        "tank_material",
        "cops_material",
        "n_stages",
        "leg_width",
        "leg_thickness",
        "leg_length",
        "displacement_0",
        "displacement_1",
        "valid",
        "score",
    ]
    para_labels = ["tank_material", "cops_material", "n_stages",
                   "leg_width", "leg_thickness", "leg_length",
                   "displacement_0", "displacement_1"]

    df = pd.read_csv(guess_csv_path, names=columns)
    # take only good and valid starting points
    df = df.loc[lambda df: df['score'] < max_score, :]

    df = theta2design(df)
    df = df.loc[lambda df: df['constraint_violation_penalty'] == 0., :]
    print(f"Evaluating from {len(df)} initial guesses.")

    df_guess = df[para_labels]

    initial_guesses = df_guess.to_numpy()

    bounds = [
        material_idxs,
        material_idxs,
        N_stages.tolist(),
        SM_feature_width.tolist(),
        SM_thickness.tolist(),
        leg_length_bounds.tolist(),
        deflection_0.tolist(),
        deflection_1.tolist()
    ]

    integrality = [
        3,
        3,
        1,
        0,
        0,
        0,
        0,
        0,
    ]

    if keep_track:
        # bad practice
        global local_opt_trials
        local_opt_trials = []
        local_opt_objective = mem_local_opt_pss_trial
    else:
        local_opt_objective = pss_trial


    for x0 in tqdm(initial_guesses):
        result = minimize(fun=local_opt_objective,
                          x0=x0,  # w, t, L, displacement_0
                          method="SLSQP",
                          bounds=bounds,
                          # constraints=None,
                          options={'maxiter': local_iter})
        # print(result.message)
        # print(result.x)

    if keep_track:
        local_opt_trials = np.array(local_opt_trials)
        df_local_trajectories = pd.DataFrame(np.arange(len(local_opt_trials)), columns=["process_id"])

        para_labels.append("score")
        df_local_trajectories[para_labels] = local_opt_trials
        # columns = [
        #     "process_id",
        #     # ...
        #     "valid",
        #     "score",
        # ]

        df_local_trajectories["valid"] = np.ones(len(df_local_trajectories))

        df_local_trajectories = df_local_trajectories[columns]

        df_local_trajectories[["tank_material", "cops_material", "n_stages"]] = df_local_trajectories[["tank_material", "cops_material", "n_stages"]].round()

        df_local_trajectories.to_csv("MIP_EGO_results_long_combined_local_opt.csv", header=False)



# </editor-fold>

if __name__ == '__main__':
    print("Starting...")
    # optimize_pss_mipego(adaptive_fn_eval=90000, warm_up_fn_eval=6000, n_cores=30,
    #                     n_per_iter=90, f_name="MIP_EGO_results_full_test.csv")

    optimize_from_guess(guess_csv_path="MIP_EGO_results_long_combined.csv", local_iter=3000,
                        max_score=30.0)






