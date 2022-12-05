import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, minimize, shgo
from matplotlib.colors import BoundaryNorm
import matplotlib.colors as colors
from matplotlib import cm
from materials_library import *
from warnings import warn
from itertools import product
from tqdm import tqdm
from copy import deepcopy
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval, STATUS_FAIL
import optunity
import pickle

np.seterr(all='raise')


class CopsTank(object):
    def __init__(self, leg_width, leg_thickness, leg_length, displacement_0,
                 cops_material, tank_material,
                 V_propellant=0.000005, V_margin=0.,
                 P_tank_max=550000, SF_ult_tank=2., SF_yield_tank=1.6,
                 P_min=100000, P_max=500000,
                 s=3, a=1, b=1,
                 n_leg_segments=3, n_stages=1,
                 K_Theta=2.65, gamma=0.85,
                 **kwargs):
        super(CopsTank, self).__init__()

        ### META
        self.__id__ = None
        self.result = None
        self.bounds = np.array([(0.2, 7.5),  # w [mm]
                               (0.1, 2.),  # t [mm]
                               (2.5, 75.),  # L [mm]
                               (-15., -0.5)]) / 1000.  # displacement_0 [mm] (input in mm -> converted to m)
        self.x0 = np.array([leg_width, leg_thickness, leg_length, displacement_0])
        self.solution = None
        self.design_converged = False
        # self. =

        ### REQUIREMENTS
        self.V_propellant = V_propellant
        self.V_margin = V_margin
        self.P_tank_max = P_tank_max
        self.P_min = P_min
        self.P_max = P_max
        self.propellant_mass = 5. / 1000.

        self.F_min_req = np.nan
        self.F_max_req = np.nan

        ### PSS
        self.pss_mass = np.nan
        self.pss_volume = np.nan

        ### COPS
        self.cops_material = deepcopy(cops_material)
        self.leg_width = leg_width
        self.leg_thickness = leg_thickness
        self.leg_length = leg_length
        self.displacement_0 = displacement_0

        self.s = s
        self.a = a
        self.b = b
        self.n = self.s * self.a
        self.m = self.s * self.b
        self.n_leg_segments = n_leg_segments
        self.n_stages = n_stages

        self.r_approx = np.nan
        self.cops_mass = np.nan
        self.spring_constant = np.nan
        self.force_response = np.nan
        self.cops_volume = np.nan

        self.K_Theta = K_Theta
        self.gamma = gamma

        # TANK
        self.tank_material = deepcopy(tank_material)
        self.SF_yield_tank = SF_yield_tank
        self.SF_ult_tank = SF_ult_tank

        self.tank_wall_thickness = np.nan
        self.cap_thickness = np.nan
        self.cap_max_deflection = np.nan
        self.tank_a = np.nan
        self.tank_area = np.nan
        self.tank_height = np.nan
        self.tank_r_inner = np.nan
        self.tank_r_outer = np.nan
        self.cap_mass = np.nan
        self.tank_wall_mass = np.nan
        self.tank_mass = np.nan

        ### SETUP
        self.constraints = np.array([
            {"type": "ineq",
             "fun": self.cops_force_displacement_constraint,
             "args": (self.P_min, 1, "max_disp")},
            {"type": "ineq",
             "fun": self.cops_force_displacement_constraint,
             "args": (self.P_max, -1, "max_disp")},
            {"type": "ineq",
             "fun": self.cops_force_displacement_constraint,
             "args": (self.P_min, 1, "min_disp")},
            {"type": "ineq",
             "fun": self.cops_force_displacement_constraint,
             "args": (self.P_max, -1, "min_disp")},
            {"type": "ineq",
             "fun": self.cops_max_sigma_constraint,
             "args": ["max_disp"]},
            {"type": "ineq",
             "fun": self.cops_max_sigma_constraint,
             "args": ["min_disp"]}
        ])

        # TODO: sensitivity analysis
        self.cops_material.set_sigma_crit(1., 1.)
        self.tank_material.set_sigma_crit(self.SF_yield_tank, self.SF_ult_tank)
        # self.finish_design(solution=self.x0)

    def __str__(self):
        if self.solution is None:
            x = self.x0
        else:
            x = self.solution

        s = f"\n" \
            f"<====================    DESIGN SUMMARY    ====================> \n" \
            f" | Converged:    {self.design_converged}\n"\
            f" | Total mass:   {self.pss_mass* 1000:5.2f} \t [g]\n" \
            f" | Total V:      {(self.pss_volume + self.V_propellant) / 0.000001:5.2f} \t [cm3]\n" \
            f" | Max force:    {self.force_response[1][-1] :5.2f} \t [N]\n" \
            f" | COPS: \n" \
            f" | \t material:   {self.cops_material.name}\n" \
            f" | \t mass:       {self.get_cops_mass(x)* 1000:5.2f} \t [g]\n" \
            f" | \t L:          {x[2]* 1000:5.2f} \t [mm]\n" \
            f" | \t t:          {x[1]* 1000:5.2f} \t [mm]\n"\
            f" | \t w:          {x[0]* 1000:5.2f} \t [mm]\n" \
            f" | \t d0:         {x[3]* 1000:5.2f} \t [mm]\n" \
            f" | \t d1:         {self.get_cops_displacement_from_design(x)* 1000:5.2f} \t [mm]\n" \
            f" | \t k:          {self.spring_constant/ 1000:5.2f} \t [N/mm]\n" \
            f" | TANK: \n" \
            f" | \t material:   {self.tank_material.name}\n" \
            f" | \t tank_mass:  {self.tank_mass* 1000:5.2f} \t [g]\n" \
            f" | \t cap_mass:   {self.cap_mass * 1000:5.2f} \t [g]\n" \
            f" | \t wall_mass:  {self.tank_wall_mass * 1000:5.2f} \t [g]\n" \
            f" | \t R_inner:    {self.tank_r_inner* 1000:5.2f} \t [mm]\n" \
            f" | \t R_outer:    {self.tank_r_outer* 1000:5.2f} \t [mm]\n" \
            f" | \t height:     {self.tank_height* 1000:5.2f} \t [mm]\n" \
            f" | \t tank_a:     {self.tank_a* 1000:5.2f} \t [mm]\n" \
            f" | \t t_walls:    {self.tank_wall_thickness* 1000:5.2f} \t [mm]\n" \
            f" | \t t_caps:     {self.cap_thickness* 1000:5.2f} \t [mm]\n" \
            f" | \t d_caps_max: {self.cap_max_deflection* 1000:5.2f} \t [mm]\n" \
            f"<==============================================================> \n"

        return s


    def get_cops_force_displacement(self, x, displacement_1):
        leg_width, leg_thickness, leg_length, displacement_0 = x

        I = leg_width * leg_thickness ** 3 / 12
        Theta = self.a * np.sin(displacement_1 / (2 * self.gamma * leg_length))
        Theta_0 = self.a * np.sin(displacement_0 / (2 * self.gamma * leg_length))
        F = ((self.n * self.m) / (self.n + self.m)) * (
                (12 * self.K_Theta * self.cops_material.E * I * (Theta - Theta_0)) / (
                leg_length ** 2 * np.cos(Theta)))

        return F

    def get_cops_stress_displacement(self, x, displacement_1):
        leg_width, leg_thickness, leg_length, displacement_0 = x
        c = leg_thickness / 2
        Theta = self.a * np.sin(displacement_1 / (2 * self.gamma * leg_length))
        max_stress = (2 * self.K_Theta * self.cops_material.E * c * (1 - self.gamma * (1 - np.cos(Theta))) * Theta) / (
                leg_length * np.cos(Theta))
        return max_stress

    def get_cops_3_leg_footprint(self, x):
        leg_width, leg_thickness, leg_length, displacement_0 = x
        L_prime = 1.3 * leg_length + 2 * leg_width  # factor from geometry fitting
        r_approx = L_prime / (2 * np.sin(np.deg2rad(120)))
        return r_approx

    def get_cops_displacement_from_design(self, x):
        __, __, tank_height, __, __ = self.get_hex_tank_geometry(x)
        displacement_1 = tank_height / self.n_stages
        return displacement_1

    def get_pressure_force(self, x, P):
        __, __, __, tank_area, __ = self.get_hex_tank_geometry(x)
        F = tank_area * P
        return F

    def get_hex_tank_geometry(self, x):
        cops_volume = self.get_cops_volume(x)
        tank_r_inner = self.get_cops_3_leg_footprint(x)
        tank_a = tank_r_inner * (2 * np.tan(np.deg2rad(60 / 2)))
        tank_r_outer = tank_a / (2 * np.sin(np.deg2rad(60 / 2)))
        tank_height = 2 * (self.V_propellant + self.V_margin + cops_volume) / (3 * np.sqrt(3) * tank_a ** 2)
        tank_area = (self.V_propellant + self.V_margin + cops_volume) / tank_height

        return tank_r_inner, tank_r_outer, tank_height, tank_area, tank_a

    def get_hex_tank_wall_thickness(self, x):
        tank_r_inner, __, __, __, __ = self.get_hex_tank_geometry(x)
        alpha = np.deg2rad(360 / (2 * 6))
        t = (+ np.sqrt(
            9 * self.P_tank_max ** 2 * tank_r_inner ** 2 + 12 * self.P_tank_max * tank_r_inner ** 2 * self.tank_material.sigma_max * np.tan(
                alpha) ** 2)
             + 3 * self.P_tank_max * tank_r_inner) / (
                    6 * self.tank_material.sigma_max)
        return np.clip(t, a_min=0.25 / 1000., a_max=None)

    def get_hex_tank_cap_thickness(self, x):
        __, tank_r_outer, __, __, __ = self.get_hex_tank_geometry(x)
        # min thickness at edges:
        t = np.sqrt((3 * self.P_tank_max * tank_r_outer ** 2) / (4 * self.tank_material.sigma_max))

        t = np.clip(t, a_min=0.5 / 1000., a_max=None)

        # TODO should be less than thickness
        deflection_center = 0.171 * self.P_tank_max * tank_r_outer ** 4 / (self.tank_material.E * t ** 3)  # approx

        return t, deflection_center

    def get_tank_volume(self,x ):
        tank_r_inner, tank_r_outer, tank_height, tank_area, tank_a = self.get_hex_tank_geometry(x)

        tank_wall_thickness = self.get_hex_tank_wall_thickness(x)
        cap_thickness, __ = self.get_hex_tank_cap_thickness(x)

        V_walls = 6 * tank_a * tank_wall_thickness * tank_height

        V_caps = 2 * 3 * tank_a ** 2 / (2 * np.tan(np.deg2rad(30))) * cap_thickness
        V_tot = V_walls + V_caps

        return V_tot, V_caps, V_walls

    def get_tank_mass(self, x):
        V_tot, V_caps, V_walls = self.get_tank_volume(x)
        tank_mass = V_tot * self.tank_material.rho
        tank_wall_mass = V_walls * self.tank_material.rho
        cap_mass = V_caps * self.tank_material.rho

        return tank_mass, tank_wall_mass, cap_mass

    def get_cops_volume(self, x):
        leg_width, leg_thickness, leg_length, displacement_0 = x
        cops_volume = leg_width * leg_thickness * leg_length \
                      * self.n_leg_segments * 3 * self.n_stages
        return cops_volume

    def get_cops_mass(self, x):
        cops_volume = self.get_cops_volume(x)
        cops_mass = cops_volume * self.cops_material.rho
        return cops_mass

    def get_pss_mass(self, x):
        tank_mass, __, __ = self.get_tank_mass(x)
        cops_mass = self.get_cops_mass(x)

        pss_mass = tank_mass + cops_mass
        return pss_mass

    def get_pss_volume(self, x):
        cops_volume = self.get_cops_volume(x)
        tank_volume, __, __ = self.get_tank_volume(x)
        return cops_volume + tank_volume

    def update_design(self, x=None):

        if x is None:
            if self.solution is None:
                raise ValueError("The design is not finished. "
                                 "If you want to manually evaluate a design you must supply it as 'solution'.")
            else:
                x = self.solution

        self.leg_width, self.leg_thickness, self.leg_length, self.displacement_0 = x

        self.tank_r_inner, self.tank_r_outer, self.tank_height, self.tank_area, self.tank_a = self.get_hex_tank_geometry(
            x)
        self.tank_wall_thickness = self.get_hex_tank_wall_thickness(x)
        self.cap_thickness, self.cap_max_deflection = self.get_hex_tank_cap_thickness(x)

        self.tank_mass, self.tank_wall_mass, self.cap_mass = self.get_tank_mass(x)

        self.pss_mass = self.get_pss_mass(x)
        self.pss_volume = self.get_pss_volume(x)

        self.F_min_req = self.get_pressure_force(x, self.P_min)
        self.F_max_req = self.get_pressure_force(x, self.P_max)

        displacement_1_trial = np.linspace(self.displacement_0,
                                           self.get_cops_displacement_from_design(x),
                                           100)
        self.force_response = self.get_cops_force_displacement(x, displacement_1_trial)

        self.spring_constant = (np.max(self.force_response) - np.min(self.force_response)) \
                               / (np.max(displacement_1_trial) - np.min(displacement_1_trial))

        self.force_response = np.array([
            displacement_1_trial,
            self.force_response
        ])

    def cops_force_displacement_constraint(self, x, P, minmax_type=1, disp_type="max_disp"):

        if disp_type == "max_disp":
            displacement_1 = self.get_cops_displacement_from_design(x)
        elif disp_type == "min_disp":
            displacement_1 = 0
        else:
            raise NotImplementedError("Must specify min or max displacement type.")

        force_actual = self.get_cops_force_displacement(x, displacement_1)
        force_required = self.get_pressure_force(x, P)

        return minmax_type * force_actual - minmax_type * force_required

    def cops_max_sigma_constraint(self, x, disp_type="max_disp"):
        if disp_type == "max_disp":
            displacement = self.get_cops_displacement_from_design(x)
        elif disp_type == "min_disp":
            displacement = np.abs(x[3])
        else:
            raise NotImplementedError("Must specify min or max displacement type.")

        max_stress_actual = self.get_cops_stress_displacement(x, displacement)
        return self.cops_material.sigma_max - max_stress_actual

    def get_cops_prod_cost(self, x):
        # TODO: sensitivity analysis
        leg_width, leg_thickness, leg_length, displacement_0 = x
        w_softmin, w_softmax = 1. / 1000., 5 / 1000.
        t_softmin, t_softmax = 0.25 / 1000., 3. / 1000.
        L_softmin, L_softmax = 10. / 1000., 30. / 1000.
        displacement_0_softmin, displacement_0_softmax = -0.1 / 1000, - 2.5 * self.get_cops_displacement_from_design(x)

        N_softmax = 5

        # t_wall_softmin = 0.5 / 1000.
        # t_caps_softmin = 0.5 / 1000.


        w_cost = 1 / np.clip(leg_width / w_softmin, a_min=None, a_max=1.) \
                 * np.clip(leg_width / w_softmax, a_min=1., a_max=None)
        t_cost = 1 / np.clip(leg_thickness / t_softmin, a_min=None, a_max=1.) \
                 * np.clip(leg_thickness / t_softmax, a_min=1., a_max=None)
        L_cost = 1 / np.clip(leg_length / L_softmin, a_min=None, a_max=1.) \
                 * np.clip(leg_length / L_softmax, a_min=1., a_max=None)
        displacement_0_cost = 1 / np.clip(displacement_0 / displacement_0_softmin, a_min=None, a_max=1.) \
                              * np.clip(displacement_0 / displacement_0_softmax, a_min=1., a_max=None)

        # t_wall_cost = 1 / np.clip(self.get_hex_tank_wall_thickness(x) / t_wall_softmin, a_min=None,
        #                              a_max=1.)

        # t_caps, __ = self.get_hex_tank_cap_thickness(x)
        # t_caps_cost = 1 / np.clip(t_caps / t_caps_softmin, a_min=None,
        #                           a_max=1.)

        N_cost = np.clip(self.n_stages / N_softmax, a_min=1., a_max=None)

        cost = (1 * 1 * w_cost * t_cost * L_cost * displacement_0_cost * N_cost) ** 2
        return cost, w_cost, t_cost, L_cost, displacement_0_cost

    def get_cops_sf_cost(self, x):
        # TODO: sensitivity analysis
        leg_width, leg_thickness, leg_length, displacement_0 = x
        sf_stress_softmin = 2.5
        sf_min_force_softmin, sf_max_force_softmax = 2.5, 0.8  # underestimation of stiffness
        t_stress_softmin, t_stress_softmax = 0.1 / 1000., 1. / 1000.

        displacement_1 = self.get_cops_displacement_from_design(x)
        max_sigma_actual = self.get_cops_stress_displacement(x, displacement_1)

        max_F_req = self.get_pressure_force(x, self.P_max)
        min_F_req = self.get_pressure_force(x, self.P_min)

        max_ext_max_F_actual = self.get_cops_force_displacement(x, displacement_1)
        min_ext_min_F_actual = self.get_cops_force_displacement(x, 0.)

        min_ext_min_F_actual / min_F_req

        sf_stress_cost = 1 / np.clip(self.cops_material.sigma_yield / max_sigma_actual / sf_stress_softmin, a_min=None, a_max=1.)
        sf_max_force_cost = np.clip(max_ext_max_F_actual / max_F_req / sf_max_force_softmax, a_min=1., a_max=None)
        sf_min_force_cost = 1 / np.clip(min_ext_min_F_actual / min_F_req / sf_min_force_softmin, a_min=None, a_max=1.)

        t_stress_cost = 1 / np.clip(leg_thickness / t_stress_softmin, a_min=None, a_max=1.) \
                 * np.clip(leg_thickness / t_stress_softmax, a_min=1., a_max=None)

        cost = (sf_stress_cost * sf_max_force_cost * sf_min_force_cost * t_stress_cost) ** 2

        return cost, sf_stress_cost, sf_max_force_cost, sf_min_force_cost

    def mass_objective(self, x):
        prod_cost, __, __, __, __ = self.get_cops_prod_cost(x)
        sf_cost, __, __, __ = self.get_cops_sf_cost(x)

        pss_mass = self.get_pss_mass(x)

        return pss_mass * prod_cost * sf_cost + self.propellant_mass

    def objective(self, x):
        mass_objective = self.mass_objective(self, x)
        # assume v_ex = 1000 m/s
        delta_v = 1000 * np.log((mass_objective) / (mass_objective - self.propellant_mass))


    def objective_cost(self, x):
        prod_cost, __, __, __, __ = self.get_cops_prod_cost(x)
        sf_cost, __, __, __ = self.get_cops_sf_cost(x)

        pss_mass = self.get_pss_mass(x)

        return pss_mass + self.propellant_mass, prod_cost * sf_cost

    def eval_constraints(self, x):
        compliance = np.full(len(self.constraints), fill_value=False)

        for i, constraint in enumerate(self.constraints):
            compliance[i] = (constraint["fun"](x, *constraint["args"]) > 0)

        return np.all(compliance)

    def _optimize_design_SMBO(self, x0, verbose=False):
        self.result = minimize(fun=self.objective,
                               x0=x0,  # w, t, L, displacement_0
                               method="SLSQP",
                               bounds=self.bounds,
                               constraints=self.constraints,
                               options={'maxiter': 500})

        # print(self.result.success)
        self.design_converged = self.result.success
        self.solution = self.result.x
        self.update_design(x=self.solution)
        return self.solution


    def optimize_design(self, x0=None, bounds=None, verbose=False):
        if bounds is None:
            bounds = np.array([(0.2, 7.5),  # w [mm]
                               (0.1, 2.),  # t [mm]
                               (2.5, 75.),  # L [mm]
                               (-15., -0.5)]) / 1000.  # displacement_0 [mm] (input in mm -> converted to m)
        self.bounds = bounds

        self.constraints = np.array([
            {"type": "ineq",
             "fun": self.cops_force_displacement_constraint,
             "args": (self.P_min, 1, "max_disp")},
            {"type": "ineq",
             "fun": self.cops_force_displacement_constraint,
             "args": (self.P_max, -1, "max_disp")},
            {"type": "ineq",
             "fun": self.cops_force_displacement_constraint,
             "args": (self.P_min, 1, "min_disp")},
            {"type": "ineq",
             "fun": self.cops_force_displacement_constraint,
             "args": (self.P_max, -1, "min_disp")},
            {"type": "ineq",
             "fun": self.cops_max_sigma_constraint,
             "args": ["max_disp"]},
            {"type": "ineq",
             "fun": self.cops_max_sigma_constraint,
             "args": ["min_disp"]}
        ])

        if x0 is None:
            x0 = np.array([self.leg_width,
                           self.leg_thickness,
                           self.leg_length,
                           self.displacement_0])
        self.x0 = x0

        self.result = minimize(fun=self.objective,
                               x0=x0,  # w, t, L, displacement_0
                               method="SLSQP",
                               bounds=bounds,
                               constraints=self.constraints,
                               options={'maxiter': 1000})

        if not self.result.success:
            warn(message="<===       WARNING       ===>\n"
                         " | Design is not converged |\n"
                         "<==========++===============>")
        self.design_converged = self.result.success

        self.solution = self.result.x

        self.update_design(x=self.solution)

        if verbose:
            print(self)

    def show_design(self, n_trial=100, zoom=0.25, refine=True):
        if self.solution is None:
            solution = self.x0
        else:
            solution = self.solution

        self.update_design(x=solution)

        if refine:
            bounds = np.array([(np.min([par * (1. - zoom), par * (1. + zoom)]),
                                np.max([par * (1. - zoom), par * (1. + zoom)]))
                               for par in solution])
        else:
            bounds = self.bounds

        w_trial = np.linspace(bounds[0][0], bounds[0][1], n_trial)
        t_trial = np.linspace(bounds[1][0], bounds[1][1], n_trial)

        ww_trial, tt_trial = np.meshgrid(w_trial, t_trial)

        L_trial = np.full_like(ww_trial, fill_value=solution[2])
        displacement_0_trial = np.full_like(ww_trial, fill_value=solution[3])

        x = np.array([ww_trial, tt_trial, L_trial, displacement_0_trial])

        displacement_1 = self.get_cops_displacement_from_design(solution)

        displacement_1_trial = np.linspace(0., displacement_1, n_trial)

        max_ext_max_F = self.cops_force_displacement_constraint(x, self.P_max, minmax_type=-1, disp_type="max_disp")
        max_ext_min_F = self.cops_force_displacement_constraint(x, self.P_min, minmax_type=1, disp_type="max_disp")
        min_ext_max_F = self.cops_force_displacement_constraint(x, self.P_max, minmax_type=-1, disp_type="min_disp")
        min_ext_min_F = self.cops_force_displacement_constraint(x, self.P_min, minmax_type=1, disp_type="min_disp")
        stress_compliance = self.cops_max_sigma_constraint(x, disp_type="max_disp")

        stress_actual = self.get_cops_stress_displacement(x, displacement_1)
        safety_factor = self.cops_material.sigma_max / stress_actual

        trust_region = np.copy(stress_actual)
        trust_region[~((max_ext_max_F > -0.05 * self.F_max_req) & (min_ext_min_F > - 0.05 * self.F_min_req))] = np.nan

        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8)) = plt.subplots(4, 2, constrained_layout=True,
                                                                             sharex=False,
                                                                             sharey=False,
                                                                             figsize=(12, 12))

        plt.suptitle(f"COPS Design:\n"
                     f"N  = {self.n_stages}, "
                     f"L  = {solution[2] * 1000:.1f} mm, "  # \n"
                     f"w  = {solution[0] * 1000:.1f} mm, "  # \n"
                     f"t  = {solution[1] * 1000:.2f} mm, "  # \n"
                     f"d0 = {solution[3] * 1000:.1f} mm, "  # \n"
                     f"d1 = {displacement_1 * 1000:.1f} mm, "  # \n"
                     f"m  = {self.pss_mass * 1000.:.1f} g",
                     weight="bold")

        norm_stress = make_color_norm(stress_compliance)

        h = ax1.contourf(w_trial * 1000., t_trial * 1000., stress_compliance, cmap="RdYlGn", norm=norm_stress)
        plt.colorbar(h, ax=ax1, label=r"$\Delta \sigma$ [Pa]")
        ax1.scatter(solution[0] * 1000, solution[1] * 1000, marker="x", color="black")
        ax1.set_ylabel(r"$t_{leg}$ [mm]")
        ax1.set_xlabel(r"$w_{leg}$ [mm]")
        ax1.set_title(r"Compliance: $\sigma_{max}$")

        k = self.spring_constant / 1000

        ax2.plot(self.force_response[0] * 1000, self.force_response[1],
                 label=r"$F_{actual}$"rf"($k \approx {k:.1f}$ [N/mm])")
        ax2.scatter([0. * 1000], [self.F_min_req], marker="x", color="black")
        ax2.scatter([displacement_1_trial[-1] * 1000], [self.F_max_req], marker="x", color="black")
        ax2.axhline(self.F_min_req, color="gray", ls="dashed", label=r"$\left( F_{req}\right)_{min,max}$")
        ax2.axhline(self.F_max_req, color="gray", ls="dashed")

        ax2.axvline(0., color="gray", ls="dotted", label=r"Operational range")
        ax2.axvline(displacement_1 * 1000, color="gray", ls="dotted")

        ax2.set_xlabel(r"$\delta_{plat}$ [mm]")
        ax2.set_ylabel("Force [N]")
        ax2.set_title("Force response.")
        ax2.legend()

        norm_max_ext_max_F = make_color_norm(max_ext_max_F)
        h = ax3.contourf(w_trial * 1000., t_trial * 1000., max_ext_max_F, cmap="RdYlGn", norm=norm_max_ext_max_F)
        plt.colorbar(h, ax=ax3, label=r"$\Delta F$ [N]")
        ax3.scatter(solution[0] * 1000, solution[1] * 1000, marker="x", color="black")
        ax3.set_ylabel(r"$t_{leg}$ [mm]")
        ax3.set_title(r"Compliance: $F_{max}(\delta_{max})$")
        ax3.set_xlabel(r"$w_{leg}$ [mm]")

        norm_max_ext_min_F = make_color_norm(max_ext_min_F)
        h = ax4.contourf(w_trial * 1000., t_trial * 1000., max_ext_min_F, cmap="RdYlGn", norm=norm_max_ext_min_F)
        plt.colorbar(h, ax=ax4, label=r"$\Delta F$ [N]")
        ax4.scatter(solution[0] * 1000, solution[1] * 1000, marker="x", color="black")
        ax4.set_title(r"Compliance: $F_{min}(\delta_{max})$")
        ax4.set_ylabel(r"$t_{leg}$ [mm]")
        ax4.set_xlabel(r"$w_{leg}$ [mm]")

        norm_min_ext_max_F = make_color_norm(min_ext_max_F)
        h = ax5.contourf(w_trial * 1000., t_trial * 1000., min_ext_max_F, cmap="RdYlGn", norm=norm_min_ext_max_F)
        plt.colorbar(h, ax=ax5, label=r"$\Delta F$ [N]")
        ax5.scatter(solution[0] * 1000, solution[1] * 1000, marker="x", color="black")
        ax5.set_ylabel(r"$t_{leg}$ [mm]")
        ax5.set_xlabel(r"$w_{leg}$ [mm]")
        ax5.set_title(r"Compliance: $F_{max}(\delta_{min})$")

        norm_min_ext_min_F = make_color_norm(min_ext_min_F)
        h = ax6.contourf(w_trial * 1000., t_trial * 1000., min_ext_min_F, cmap="RdYlGn", norm=norm_min_ext_min_F)
        plt.colorbar(h, ax=ax6, label=r"$\Delta F$ [N]")
        ax6.scatter(solution[0] * 1000, solution[1] * 1000, marker="x", color="black")
        ax6.set_title(r"Compliance: $F_{min}(\delta_{min})$")
        ax6.set_ylabel(r"$t_{leg}$ [mm]")
        ax6.set_xlabel(r"$w_{leg}$ [mm]")

        norm_trust_region = make_color_norm(trust_region)
        h = ax7.contourf(w_trial * 1000., t_trial * 1000., trust_region, cmap="RdYlGn", norm=norm_trust_region)
        plt.colorbar(h, ax=ax7, label=r"$\sigma_{max}$ [Pa]")
        ax7.scatter(solution[0] * 1000, solution[1] * 1000, marker="x", color="black")
        ax7.set_ylabel(r"$t_{leg}$ [mm]")
        ax7.set_xlabel(r"$w_{leg}$ [mm]")
        ax7.set_title(r"Stress trust region ($0.95 \cdot F_{req} < F_{actual} < 1.05 \cdot F_{req}$)")

        sf_max = 5.
        sf_v_max = np.minimum(np.nanmax(safety_factor), sf_max)
        norm_sf = colors.TwoSlopeNorm(vmin=0., vcenter=1., vmax=np.maximum(sf_v_max, 1.2))
        h = ax8.contourf(w_trial * 1000., t_trial * 1000., safety_factor, cmap="RdYlGn", norm=norm_sf,
                         vmin=0., vmax=sf_v_max, levels=10)
        plt.colorbar(h, ax=ax8, label=r"SF [-]")
        ax8.scatter(solution[0] * 1000, solution[1] * 1000, marker="x", color="black")
        ax8.set_ylabel(r"$t_{leg}$ [mm]")
        ax8.set_xlabel(r"$w_{leg}$ [mm]")
        ax8.set_title(r"Safety factor ($\sigma_{max}$).")

        plt.show()


def make_color_norm(arr):
    try:
        norm = colors.TwoSlopeNorm(vmin=np.nanmin(arr), vcenter=0., vmax=np.nanmax(arr))
    except ValueError:
        delta = np.nanmax(arr) - np.nanmin(arr)
        try:
            norm = colors.TwoSlopeNorm(vmin=np.nanmin(arr) - 2 * delta / 10,
                                       vcenter=np.nanmin(arr) - delta / 10,
                                       vmax=np.nanmax(arr))
        except ValueError:
            try:

                norm = colors.TwoSlopeNorm(vmin=np.nanmin(arr),
                                           vcenter=np.nanmax(arr) + delta / 10,
                                           vmax=np.nanmax(arr) + 2 * delta / 10)
            except ValueError:
                return None

    return norm

def rank_designs(arr: np.array([], dtype=CopsTank)):

    return sorted(arr, key=lambda x: x.pss_mass)

def sweep_designs(verbose=False, n_best=1):
    x0 = np.array([4., 1., 25., -5.]) / 1000.
    leg_width, leg_thickness, leg_length, displacement_0 = x0

    materials = np.array([Ti6Al4V, A7075T6, SST301FH, SST304])

    candidate_material_pairs = np.array([prod for prod in product(materials, repeat=2)])

    candidates = np.full(shape=len(candidate_material_pairs), fill_value=object, dtype=object)

    for i, candidate_materials in tqdm(enumerate(candidate_material_pairs)):
        cops_material = deepcopy(candidate_materials[0])
        tank_material = deepcopy(candidate_materials[1])
        candidate = CopsTank(leg_width=leg_width, leg_thickness=leg_thickness,
                             leg_length=leg_length, displacement_0=displacement_0,
                             cops_material=cops_material, tank_material=tank_material,
                             n_stages=2)

        # candidate.show_design()
        candidate.optimize_design(verbose=False)
        # candidate.show_design()

        candidates[i] = candidate

    candidates_sorted = rank_designs(candidates)

    if verbose:
        n_best = -1

    print(f"\n"
          f"<=======================>\n"
          f"<=       RESULTS       =>\n"
          f"<=      {n_best:2.0f} best        =>\n"
          f"<=======================>\n")

    for i, c in enumerate(candidates_sorted[:n_best]):
        print(f"<||||||||| Rank {i+1} |||||||||>")
        print(c)

    candidates_sorted[0].show_design()


def objective_general(leg_width, leg_thickness, leg_length, displacement_0, tank_material, cops_material, n_stages):

    candidate = CopsTank(leg_width=leg_width, leg_thickness=leg_thickness,
                         leg_length=leg_length, displacement_0=displacement_0,
                         cops_material=tank_material, tank_material=cops_material,
                         n_stages=n_stages)

    x0 = np.array([leg_width,
                   leg_thickness,
                   leg_length,
                   displacement_0])

    return candidate.objective(x0)

def objective_SMBO(vector, use_local_opt=True):
    # try:
    x0 = vector["leg_width"], vector["leg_thickness"], vector["leg_length"], vector["displacement_0"]
    candidate = CopsTank(**vector)

    if use_local_opt:
        x1 = candidate._optimize_design_SMBO(x0)
    else:
        x1 = np.zeros(4)

    mass, cost = candidate.objective_cost(x0)

    compliance = candidate.eval_constraints(x0)

    if candidate.design_converged:
        status = STATUS_OK
    elif compliance:
        status = STATUS_OK
    else:
        status = STATUS_FAIL

    tank_r_inner, tank_r_outer, tank_height, tank_area, tank_a = candidate.get_hex_tank_geometry(x1)

    return {'loss': mass * (1 + np.log(cost)), 'status': status, "attachments":
        {"x1": x1, "mass": mass, "cost": cost, "F_min": candidate.get_cops_force_displacement(x1, 0),
         "F_max": candidate.get_cops_force_displacement(x1, candidate.get_cops_displacement_from_design(x1)),
         "displacement": candidate.get_cops_displacement_from_design(x1),
         "tank_height": tank_height, "tank_r_outer": tank_r_outer, "tank_a": tank_a, "tank_area": tank_area,
         "F_min_req": candidate.get_pressure_force(x1, candidate.P_min),
         "F_max_req": candidate.get_pressure_force(x1, candidate.P_max)}}


def sweep_SMBO():
    trials = Trials()

    space = {
        "leg_width": hp.uniform("leg_width", 0.2 / 1000, 7.5 / 1000),
        "leg_thickness": hp.uniform("leg_thickness", 0.1 / 1000, 2. / 1000),
        "leg_length": hp.uniform("leg_length", 2.5 / 1000, 75. / 1000),
        "displacement_0": hp.uniform("displacement_0", -15. / 1000, -0.5 / 1000),
        "tank_material": hp.choice("tank_material", materials_list),
        "cops_material": hp.choice("cops_material", materials_list),
        "n_stages": hp.quniform("n_stages", 1, 20, 1)
    }

    tpe._default_n_startup_jobs = 4000

    with np.errstate(under='ignore'):
        best_param = fmin(
            fn=objective_SMBO,
            space=space,
            algo=tpe.suggest,
            max_evals=5000,
            trials=trials
        )

    from hyperopt.plotting import main_plot_history, main_plot_histogram, main_plot_vars

    main_plot_history(trials)
    main_plot_histogram(trials)
    main_plot_vars(trials)

    # space_eval(space, best_param)

    best_param["cops_material"] = materials_list[best_param["cops_material"]]
    best_param["tank_material"] = materials_list[best_param["tank_material"]]

    x1 = best_param["leg_width"], best_param["leg_thickness"], best_param["leg_length"], best_param["displacement_0"]

    candidate = CopsTank(**best_param)
    candidate.optimize_design(x1)

    candidate.show_design()

    print(candidate)


    # print(trials.trials)

    with open("hyperopt_new", "wb") as f:
        pickle.dump(trials, f)



def design():
    x0 = np.array([4., 1., 25., -5.]) / 1000.
    leg_width, leg_thickness, leg_length, displacement_0 = x0

    candidate = CopsTank(leg_width=leg_width, leg_thickness=leg_thickness,
                         leg_length=leg_length, displacement_0=displacement_0,
                         cops_material=Ti6Al4V, tank_material=A7075T6,
                         n_stages=2)


    candidate.optimize_design(verbose=False)



    print(f"\n"
          f"<=======================>\n"
          f"<=       RESULTS       =>\n"
          f"<=======================>\n")

    print(candidate)

    candidate.show_design()

if __name__ == '__main__':
    # sweep_designs(verbose=True)
    # design()
    sweep_SMBO()
