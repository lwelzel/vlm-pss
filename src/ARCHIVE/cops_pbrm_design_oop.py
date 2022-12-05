import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, minimize, shgo
from matplotlib.colors import BoundaryNorm
import matplotlib.colors as colors
from matplotlib import cm

np.seterr(all='raise')


class Material(object):
    def __init__(self, E, rho, sigma_max, sigma_yield,
                 name,
                 **kwargs):
        super(Material, self).__init__()

        self.name = name
        self.E = E
        self.rho = rho
        self.sigma_max = sigma_max
        self.sigma_yield = sigma_yield


class CopsTank(object):
    def __init__(self, leg_width, leg_thickness, leg_length, displacement_0,
                 cops_material, tank_material,
                 V_tank=0.000005, V_margin=0.,
                 P_tank_max=550000,
                 P_min=100000, P_max=500000,
                 s=3, a=1, b=1,
                 n_leg_segments=3, n_stages=1,
                 K_Theta=2.65, gamma=0.85,
                 **kwargs):
        super(CopsTank, self).__init__()

        ### META
        self.__id__ = None
        self.result = None
        self.bounds = None
        self.constraints = None
        self.x0 = np.array([leg_width, leg_thickness, leg_length, displacement_0])
        self.solution = None
        # self. =

        ### REQUIREMENTS
        self.V_tank = V_tank
        self.V_margin = V_margin
        self.P_tank_max = P_tank_max
        self.P_min = P_min
        self.P_max = P_max
        self.propellant_mass = 5. / 1000.

        self.F_min_req = None
        self.F_max_req = None

        ### PSS
        self.pss_mass = 1.  # np.nan

        ### COPS
        self.cops_material = cops_material
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

        self.r_approx = 1.  # np.nan
        self.cops_mass = 1.  # np.nan
        self.spring_constant = 1.  # np.nan
        self.force_response = 1.  # np.nan
        self.cops_volume = 1.  # np.nan

        self.K_Theta = K_Theta
        self.gamma = gamma

        # TANK
        self.tank_material = tank_material

        self.tank_wall_thickness = 1.  # np.nan
        self.cap_thickness = 1.  # np.nan
        self.cap_max_deflection = 1.  # np.nan
        self.tank_a = 1.  # np.nan
        self.tank_area = 1.  # np.nan
        self.tank_height = 1.  # np.nan
        self.tank_r_inner = 1.  # np.nan
        self.tank_r_outer = 1.  # np.nan
        self.cap_mass = 1.  # np.nan
        self.tank_wall_mass = 1.  # np.nan
        self.tank_mass = 1.  # np.nan

        ### SETUP
        print("pre update")
        self.update_design(self.x0)  # initialize design
        print("pre finish")
        # self.finish_design(solution=self.x0)
        # print("post finish")

    def get_cops_force_displacement(self, displacement_1):
        I = self.leg_width * self.leg_thickness ** 3 / 12
        Theta = self.a * np.sin(displacement_1 / (2 * self.gamma * self.leg_length))
        Theta_0 = self.a * np.sin(self.displacement_0 / (2 * self.gamma * self.leg_length))
        F = ((self.n * self.m) / (self.n + self.m)) * (
                (12 * self.K_Theta * self.cops_material.E * I * (Theta - Theta_0)) / (
                self.leg_length ** 2 * np.cos(Theta)))
        print(f"F: {F}")
        return F

    def get_cops_stress_displacement(self, displacement_1):
        c = self.leg_thickness / 2
        Theta = self.a * np.sin(displacement_1 / (2 * self.gamma * self.leg_length))
        max_stress = (2 * self.K_Theta * self.cops_material.E * c * (1 - self.gamma * (1 - np.cos(Theta))) * Theta) / (
                self.leg_length * np.cos(Theta))
        print(f"max_stress: {max_stress}")
        return max_stress

    def get_cops_3_leg_footprint(self):
        L_prime = 1.3 * self.leg_length + 2 * self.leg_width  # factor from geometry fitting
        r_approx = L_prime / (2 * np.sin(np.deg2rad(120)))
        return r_approx

    def get_cops_displacement_from_design(self):
        displacement_1 = self.tank_height / self.n_stages
        return displacement_1

    def get_pressure_force(self, P):
        F = self.tank_area * P
        return F

    def set_hex_tank_geometry(self):
        self.set_cops_volume()
        self.tank_r_inner = self.get_cops_3_leg_footprint()
        self.tank_a = self.tank_r_inner * (2 * np.tan(np.deg2rad(60 / 2)))
        self.tank_r_outer = self.tank_a / (2 * np.sin(np.deg2rad(60 / 2)))
        self.tank_height = 2 * (self.V_tank + self.V_margin + self.cops_volume) / (3 * np.sqrt(3) * self.tank_a ** 2)
        self.tank_area = self.V_tank / self.tank_height

    def set_hex_tank_wall_thickness(self):
        alpha = np.deg2rad(360 / (2 * 6))
        t = (+ np.sqrt(
            9 * self.P_tank_max ** 2 * self.tank_r_inner ** 2 + 12 * self.P_tank_max * self.tank_r_inner ** 2 * self.tank_material.sigma_max * np.tan(
                alpha) ** 2)
             + 3 * self.P_tank_max * self.tank_r_inner) / (
                    6 * self.tank_material.sigma_max)
        self.tank_wall_thickness = t

    def set_hex_tank_cap_thickness(self):
        # min thickness at edges:
        t = np.sqrt((3 * self.P_tank_max * self.tank_r_outer ** 2) / (4 * self.tank_material.sigma_max))

        # TODO should be less than thickness
        deflection_center = 0.171 * self.P_tank_max * self.tank_r_outer ** 4 / (self.tank_material.E * t ** 3)  # approx

        self.cap_thickness = t
        self.cap_max_deflection = deflection_center

    def set_mass_hex_tank(self):
        V_walls = 6 * self.tank_a * self.tank_wall_thickness * self.tank_height

        V_caps = 2 * 3 * self.tank_a ** 2 / (2 * np.tan(np.deg2rad(30))) * self.cap_thickness
        V_tot = V_walls + V_caps

        self.tank_mass = V_tot * self.tank_material.rho
        self.tank_wall_mass = V_walls * self.tank_material.rho
        self.cap_mass = V_caps * self.tank_material.rho

    def set_tank_mass(self):
        self.set_hex_tank_wall_thickness()
        self.set_hex_tank_cap_thickness()
        self.set_mass_hex_tank()

    def set_cops_volume(self):
        self.cops_volume = self.leg_width * self.leg_thickness * self.leg_length \
                           * self.n_leg_segments * 3 * self.n_stages

    def set_cops_mass(self):
        self.set_cops_volume()
        self.cops_mass = self.cops_volume * self.cops_material.rho

    def set_pss_mass(self):
        self.set_tank_mass()
        self.set_cops_mass()

        self.pss_mass = self.tank_mass + self.cops_mass

    def update_design(self, x):
        self.leg_width, self.leg_thickness, self.leg_length, self.displacement_0 = x

        self.set_hex_tank_geometry()
        self.set_hex_tank_wall_thickness()
        self.set_hex_tank_cap_thickness()

        self.set_pss_mass()

        # print(self.pss_mass)

        # print("--->")

    def finish_design(self, solution=None):
        if solution is None:
            if self.solution is None:
                raise ValueError("The design is not finished. "
                                 "If you want to manually evaluate a design you must supply it as 'solution'.")
            else:
                solution = self.solution

        self.update_design(solution)

        self.F_min_req = self.get_pressure_force(self.P_min)
        self.F_max_req = self.get_pressure_force(self.P_max)

        displacement_1_trial = np.linspace(self.displacement_0,
                                           self.get_cops_displacement_from_design(),
                                           100)
        self.force_response = self.get_cops_force_displacement(displacement_1_trial)

        self.spring_constant = (np.max(self.force_response) - np.min(self.force_response)) \
                               / (np.max(displacement_1_trial) - np.min(displacement_1_trial))

        self.force_response = np.array([
            displacement_1_trial,
            self.force_response
        ])

    def cops_force_displacement_constraint(self, x, P, minmax_type=1, disp_type="max_disp"):
        self.update_design(x)

        if disp_type == "max_disp":
            displacement_1 = self.get_cops_displacement_from_design()
        elif disp_type == "min_disp":
            displacement_1 = 0
        else:
            raise NotImplementedError("Must specify min or max displacement type.")

        force_actual = self.get_cops_force_displacement(displacement_1)
        force_required = self.get_pressure_force(P)

        return minmax_type * force_actual - minmax_type * force_required

    def cops_max_sigma_constraint(self, x):
        self.update_design(x)

        displacement_1 = self.get_cops_displacement_from_design()
        max_stress_actual = self.get_cops_stress_displacement(displacement_1)
        return self.cops_material.sigma_max - max_stress_actual

    def get_cops_prod_cost(self):
        w_softmin, w_softmax = 1. / 1000., 5. / 1000.
        t_softmin, t_softmax = 0.5 / 1000., 3. / 1000.
        L_softmin, L_softmax = 10. / 1000., 25. / 1000.
        displacement_0_softmin, displacement_0_softmax = -0.5 / 1000., -5. / 1000.

        w_cost = 1 / np.clip(self.leg_width / w_softmin, a_min=None, a_max=1.) \
                 * np.clip(self.leg_width / w_softmax, a_min=1., a_max=None)
        t_cost = 1 / np.clip(self.leg_thickness / t_softmin, a_min=None, a_max=1.) \
                 * np.clip(self.leg_thickness / t_softmax, a_min=1., a_max=None)
        L_cost = 1 / np.clip(self.leg_length / L_softmin, a_min=None, a_max=1.) \
                 * np.clip(self.leg_length / L_softmax, a_min=1., a_max=None)
        displacement_0_cost = 1 / np.clip(self.displacement_0 / displacement_0_softmin, a_min=None, a_max=1.) \
                              * np.clip(self.displacement_0 / displacement_0_softmax, a_min=1., a_max=None)

        cost = 1. + (w_cost * t_cost * L_cost * displacement_0_cost - 1.) ** 1.5
        return cost

    def get_cops_sf_cost(self):
        displacement_1 = self.get_cops_displacement_from_design()
        max_sigma_actual = self.get_cops_stress_displacement(displacement_1)

    def objective(self, x):
        self.update_design(x)

        cost = self.get_cops_prod_cost()

        return self.pss_mass * cost + self.propellant_mass

    def optimize_design(self, x0=None, bounds=None, ):
        if bounds is None:
            self.bounds = np.array([(0.2, 5.),  # w [mm]
                                    (0.1, 2.),  # t [mm]
                                    (2.5, 75.),  # L [mm]
                                    (-15., -0.5)]) / 1000.  # displacement_0 [mm] (input in mm -> converted to m)
        else:
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
             "fun": self.cops_max_sigma_constraint}
        ])

        if x0 is None:
            self.x0 = np.array([self.leg_width,
                                self.leg_thickness,
                                self.leg_length,
                                self.displacement_0])
        else:
            self.x0 = x0
            self.update_design(self.x0)

        print("opt =================")
        print(self.leg_width, self.leg_thickness, self.leg_length, self.n_leg_segments, self.n_stages)

        self.result = minimize(fun=self.objective,
                               x0=x0,  # w, t, L, displacement_0
                               method="SLSQP",
                               bounds=self.bounds,
                               constraints=self.constraints,
                               options={'maxiter': 1000})

        self.solution = self.result.x

        self.finish_design(solution=self.solution)

    def __trial_cops_force_displacement(self, x, displacement_1):
        leg_width, leg_thickness, leg_length, displacement_0 = x
        I = leg_width * leg_thickness ** 3 / 12
        Theta = self.a * np.sin(displacement_1 / (2 * self.gamma * leg_length))
        Theta_0 = self.a * np.sin(displacement_0 / (2 * self.gamma * leg_length))

        return ((self.n * self.m) / (self.n + self.m)) * (
                (12 * self.K_Theta * self.cops_material.E * I * (Theta - Theta_0)) / (
                leg_length ** 2 * np.cos(Theta)))

    def __trial_cops_stress_displacement(self, x, displacement_1):
        leg_width, leg_thickness, leg_length, displacement_0 = x
        c = leg_thickness / 2
        Theta = self.a * np.sin(displacement_1 / (2 * self.gamma * leg_length))
        max_stress = (2 * self.K_Theta * self.cops_material.E * c * (1 - self.gamma * (1 - np.cos(Theta))) * Theta) / (
                leg_length * np.cos(Theta))
        return max_stress

    def show_design(self, n_trial=100, zoom=0.25, refine=True):
        if self.solution is None:
            solution = self.x0
        else:
            solution = self.solution

        self.finish_design(solution=solution)
        self.update_design(solution)

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

        displacement_1 = self.get_cops_displacement_from_design()

        displacement_1_trial = np.linspace(0., displacement_1, n_trial)

        max_ext_max_F = self.cops_force_displacement_constraint(x, self.P_max, minmax_type=-1, disp_type="max_disp")
        max_ext_min_F = self.cops_force_displacement_constraint(x, self.P_min, minmax_type=1, disp_type="max_disp")
        min_ext_max_F = self.cops_force_displacement_constraint(x, self.P_max, minmax_type=-1, disp_type="min_disp")
        min_ext_min_F = self.cops_force_displacement_constraint(x, self.P_min, minmax_type=1, disp_type="min_disp")
        stress_compliance = self.cops_max_sigma_constraint(x)

        stress_actual = self.get_cops_stress_displacement(displacement_1)
        safety_factor = self.cops_material.sigma_max / stress_actual

        self.finish_design(solution=solution)

        trust_region = np.copy(stress_actual)
        trust_region[~((max_ext_max_F > -0.05 * self.F_max_req) & (min_ext_min_F > - 0.05 * self.F_min_req))] = np.nan

        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8)) = plt.subplots(4, 2, constrained_layout=True,
                                                                             sharex=False,
                                                                             sharey=False,
                                                                             figsize=(12, 12))

        plt.suptitle(f"COPS Design:\n"
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


if __name__ == '__main__':
    titanium = Material(name="Ti",
                        E=110.0 * 10 ** 9, rho=4480, sigma_max=1.103 * 10 ** 9, sigma_yield=1.)  # np.nan)

    x0 = np.array([4., 1., 25., -5.]) / 1000.
    leg_width, leg_thickness, leg_length, displacement_0 = x0

    candidate = CopsTank(leg_width=leg_width, leg_thickness=leg_thickness,
                         leg_length=leg_length, displacement_0=displacement_0,
                         cops_material=titanium, tank_material=titanium)

    # candidate.show_design()
    candidate.optimize_design()
    # candidate.show_design()
