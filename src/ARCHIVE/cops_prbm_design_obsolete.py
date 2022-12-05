import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, minimize, shgo
from matplotlib.colors import BoundaryNorm
import matplotlib.colors as colors
from matplotlib import cm


def cops_force_displacement(x, displacement_1, E, n, m, a, k_Theta=2.65, gamma=0.85):
    w, t, L, displacement_0 = x

    I = w * t ** 3 / 12
    Theta = a * np.sin(displacement_1 / (2 * gamma * L))
    Theta_0 = a * np.sin(displacement_0 / (2 * gamma * L))

    return ((n * m) / (n + m)) * ((12 * k_Theta * E * I * (Theta - Theta_0)) / (L ** 2 * np.cos(Theta)))

def cops_stress_displacement(x, displacement_1, E, a, k_Theta=2.65, gamma=0.85):
    w, t, L, displacement_0 = x
    c = t/2
    Theta = a * np.sin(displacement_1 / (2 * gamma * L))
    max_stress = (2 * k_Theta * E * c * (1 - gamma * (1 - np.cos(Theta))) * Theta) / (L * np.cos(Theta))
    return max_stress

def cops_force_displacement_constraint(x, E, P, n, m, a, n_leg_segments, n_stages, k_Theta=2.65, gamma=0.85, minmax_type=1,
                           disp_type="max_disp"):

    if disp_type == "max_disp":
        displacement_1 = cops_displacement_from_design(x, n_stages)
    elif disp_type == "min_disp":
        displacement_1 = 0
    else:
        raise NotImplementedError("Must specify min or max displacement type.")

    force_actual = cops_force_displacement(x, displacement_1, E, n, m, a, k_Theta, gamma)
    force_required = pressure_force(x, P)

    return minmax_type * force_actual - minmax_type * force_required

def cops_max_sigma_constraint(x, E, sigma_max, a, n_stages, k_Theta=2.65, gamma=0.85):
    displacement_1 = cops_displacement_from_design(x, n_stages)
    max_stress_actual = cops_stress_displacement(x, displacement_1, E, a, k_Theta, gamma)
    return sigma_max - max_stress_actual

def cops_3_leg_footprint(x):
    w, t, L, displacement_0 = x
    L_prime = 1.3 * L + 2 * w  # factor from geometry fitting
    r_approx = L_prime / (2 * np.sin(np.deg2rad(120)))
    return r_approx

def cops_displacement_from_design(x, n_stages):
    __, __, __, h, __ = hex_tank_geometry(x)
    displacement_1 = h / n_stages

    return displacement_1

def pressure_force(x, P,):
    __, __, __, __, A = hex_tank_geometry(x)
    F = A * P
    return F

def hex_tank_geometry(x, V=0.000005, margin=0.15):
    V_req = (1 + margin) * V  # TODO: replace margin with leg volume + margin

    r_inner_approx = cops_3_leg_footprint(x)
    a = r_inner_approx * (2 * np.tan(np.deg2rad(60 / 2)))
    r_outer = a / (2 * np.sin(np.deg2rad(60 / 2)))
    h = 2 * V_req / (3 * np.sqrt(3) * a ** 2)
    A = V_req / h

    return r_inner_approx, r_outer, a, h, A


def hex_tank_wall_thickness(P, r_inner, sigma_max, n=6):
    """
    Based on discussion: https://physics.stackexchange.com/questions/104297/wall-stress-of-a-hexagonal-pressure-vessel

    r = r_inscribed
    \sigma= P\left(\frac{r}{t}+\tan (\alpha)^2 \frac{r^2}{3 t^2}\right)
    rearranged for t:
    t=\frac{3 a r \pm  \sqrt{9 a^2 r^2+12 a r^2 x \tan ^2(b)}}{6 x}

    :param P:
    :param r:
    :param n:
    :param E:
    :return:
    """

    alpha = np.deg2rad(360 / (2 * n))


    t = (+ np.sqrt(9 * P ** 2 * r_inner ** 2 + 12 * P * r_inner ** 2 * sigma_max * np.tan(alpha) ** 2) + 3 * P * r_inner) / (
                6 * sigma_max)
    # both are valid solutions, and typically both are physical.
    # However, we expect higher required wall thickness for larger pressure vessel radii -> above solution is valid
    # t = (- np.sqrt(9 * P ** 2 * r_in ** 2 + 12 * P * r_in ** 2 * sigma_max * np.tan(alpha) ** 2) + 3 * P * r_in) / (
    #         6 * sigma_max)

    return t


def hex_tank_cap_thickness(P, r_outer, E, sigma_max, SF=5):
    """
    Circular plate, edges clamped, uniform load
    Assumption: small deflection compared to wall thickness
    Take conservative r_out (circumscribed)
    :param P:
    :param r:
    :param E:
    :param SF:
    :return:
    """
    sigma_max = sigma_max / SF
    # min thickness at edges:
    t = np.sqrt((3 * P * r_outer**2) / (4 * sigma_max))

    # TODO should be less than thickness
    deflection_center = 0.171 * P * r_outer**4 / (E * t**3)  # approx

    return t


def mass_hex_tank(a, t_wall, t_caps, h, rho):
    V_walls = 6 * a * t_wall * h

    V_caps = 2 * 3 * a**2 / (2 * np.tan(np.deg2rad(30))) * t_caps
    V_tot = V_walls + V_caps

    mass_tot = V_tot * rho
    mass_walls = V_walls * rho
    mass_caps = V_caps * rho

    return mass_tot, mass_walls, mass_caps

def tank_mass_(x, sigma_max, E, rho, P=550000,):
    r_inner, r_outer, a, h, A = hex_tank_geometry(x)

    t_walls = hex_tank_wall_thickness(P, r_inner, sigma_max)
    t_caps = hex_tank_cap_thickness(P, r_outer, E, sigma_max)

    mass_tot, __, __ = mass_hex_tank(a, t_walls, t_caps, h, rho)
    return mass_tot

def cops_mass(x, rho):
    w, t, L, displacement_0 = x
    volume_legs = w * t * L * 2 * 3
    mass_legs = volume_legs * rho
    return mass_legs

def pss_mass(x, args, use_cost=True):
    w, t, L, displacement_0 = x
    rho, sigma_max, E, n_leg_segments, n_stages = args
    propellant_mass = 5. / 1000.  # g

    mass_tank = tank_mass_(x, sigma_max,E, rho)
    mass_cops = cops_mass(x, rho)

    mass_total = mass_tank + mass_cops

    cost_corr = cops_prod_cost(x)

    mass_adj = mass_total * (1. + use_cost * (cost_corr - 1.)) + propellant_mass

    return mass_adj

def cops_prod_cost(x):
    w, t, L, displacement_0 = x

    w_softmin, w_softmax = 1. / 1000., 5. / 1000.
    t_softmin, t_softmax = 0.5 / 1000., 3. / 1000.
    L_softmin, L_softmax = 10. / 1000., 25. / 1000.
    displacement_0_softmin, displacement_0_softmax = -0.5 / 1000., -5. / 1000.

    w_cost = 1 / np.clip(w / w_softmin, a_min=None, a_max=1.) * np.clip(w / w_softmax, a_min=1., a_max=None)
    t_cost = 1 / np.clip(t / t_softmin, a_min=None, a_max=1.) * np.clip(t / t_softmax, a_min=1., a_max=None)
    L_cost = 1 / np.clip(L / L_softmin, a_min=None, a_max=1.) * np.clip(L / L_softmax, a_min=1., a_max=None)
    displacement_0_cost = 1 / np.clip(displacement_0 / displacement_0_softmin, a_min=None, a_max=1.) \
                          * np.clip(displacement_0 / displacement_0_softmax, a_min=1., a_max=None)

    cost = 1. + (w_cost * t_cost * L_cost * displacement_0_cost - 1.)**1.5
    return cost

def cops_sf_cost(x, E, a, n_stages, sigma_max):
    displacement_1 = cops_displacement_from_design(x, n_stages)
    max_sigma_actual = cops_stress_displacement(x, displacement_1, E, a)


def find_opt_spring(E, rho, sigma_max, SF, s=3, a=1, b=1, k_Theta=2.65, gamma=0.85):
    n_leg_segments, n_stages = 2, 1
    x0 = np.array([4., 1., 25., -5.]) / 1000  # np.array([2., 0.5, 15., -5.]) / 1000
    bounds = np.array([(0.2, 5.),             # w [mm]
                       (0.1, 2.),              # t [mm]
                       (2.5, 75.),             # L [mm]
                       (-15., -0.5)]) / 1000.  # displacement_0 [mm] (input in mm -> converted to m)

    args = np.array([rho,
                     sigma_max,
                     E,
                     n_leg_segments,
                     n_stages])

    P_tank_max = 550000
    P_max = 500000
    P_min = 100000

    sigma_max = sigma_max / SF

    n = s * a
    m = s * b

    test_cops_design(bounds, x0, args, P_tank_max, P_max, P_min, m, n, s, a, b, n_stages, k_Theta=2.65, gamma=0.85)

    # ================================================================

    # x, E, P, n, m, a, n_leg_segments, n_stages, k_Theta=2.65, gamma=0.85, minmax_type=1, disp_type="max_disp"

    constraints = [
        {"type": "ineq",
         "fun": cops_force_displacement_constraint,
         "args": (E, P_min, n, m, a, n_leg_segments, n_stages, k_Theta, gamma, 1, "max_disp")},
        {"type": "ineq",
         "fun": cops_force_displacement_constraint,
         "args": (E, P_max, n, m, a, n_leg_segments, n_stages, k_Theta, gamma, -1, "max_disp")},
        {"type": "ineq",
         "fun": cops_force_displacement_constraint,
         "args": (E, P_min, n, m, a, n_leg_segments, n_stages, k_Theta, gamma, 1, "min_disp")},
        {"type": "ineq",
         "fun": cops_force_displacement_constraint,
         "args": (E, P_max, n, m, a, n_leg_segments, n_stages, k_Theta, gamma, -1, "min_disp")},
        {"type": "ineq",
         "fun": cops_max_sigma_constraint,
         "args": (E, sigma_max, a, n_stages, k_Theta, gamma)}
    ]


    result = minimize(fun=pss_mass,
                      x0=x0,  # w, t, L, displacement_0
                      args=args,
                      method="SLSQP",
                      bounds=bounds,
                      constraints=constraints,
                      options={'maxiter': 1000})

    # result = shgo(func=cops_mass,
    #                   args=mat_prop,
    #                   bounds=bounds,
    #                   constraints=constraints,
    #                   options={'maxiter': 1000})

    print(result)

    para = ["leg_width     ",
            "leg_thickness ",
            "leg_length    ",
            "displacement_0"]

    para_vals = result.x

    for par, val in zip(para, para_vals):
        print(f"{par} = {val * 1000.:.2f} [mm]")

    print(f"{'Mass          '} = {pss_mass(para_vals, args, use_cost=False) * 1000.:.2f} [g]")

    test_cops_design(bounds, para_vals, args, P_tank_max, P_max, P_min, m, n, s, a, b, n_stages,
                     k_Theta=2.65, gamma=0.85, refined=True)

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



def test_cops_design(bounds, x0, args, P_tank_max, P_max, P_min, m, n, s, a, b, n_stages,
                     k_Theta=2.65, gamma=0.85, refined=False):
    n_trial = 100

    if refined:
        bounds = np.array([(np.min([par * 0.75, par * 1.25]), np.max([par * 0.75, par * 1.25])) for par in x0])


    w_trial = np.linspace(bounds[0][0], bounds[0][1], n_trial)
    t_trial = np.linspace(bounds[1][0], bounds[1][1], n_trial)

    ww_trial, tt_trial = np.meshgrid(w_trial, t_trial)


    L_trial = np.full_like(ww_trial, fill_value=x0[2])
    displacement_0_trial = np.full_like(ww_trial, fill_value=x0[3])

    x = np.array([ww_trial, tt_trial, L_trial, displacement_0_trial])

    rho, sigma_max, E, n_leg_segments, n_stages = args

    displacement_1 = cops_displacement_from_design(x0, n_stages=1.)

    displacement_1_trial = np.linspace(0., displacement_1, n_trial)

    max_ext_max_F = cops_force_displacement_constraint(x, E, P_max, n, m, a, n_leg_segments, n_stages, k_Theta, gamma,
                                           minmax_type=-1, disp_type="max_disp")
    max_ext_min_F = cops_force_displacement_constraint(x, E, P_min, n, m, a, n_leg_segments, n_stages, k_Theta, gamma,
                                           minmax_type=1, disp_type="max_disp")
    min_ext_max_F = cops_force_displacement_constraint(x, E, P_max, n, m, a, n_leg_segments, n_stages, k_Theta, gamma,
                                           minmax_type=-1, disp_type="min_disp")
    min_ext_min_F = cops_force_displacement_constraint(x, E, P_min, n, m, a, n_leg_segments, n_stages, k_Theta, gamma,
                                           minmax_type=1, disp_type="min_disp")
    stress_compliance = cops_max_sigma_constraint(x, E, sigma_max, a, n_stages, k_Theta, gamma)
    stress_actual = cops_stress_displacement(x, displacement_1, E, a, k_Theta, gamma)
    safety_factor = sigma_max / stress_actual

    force_response = cops_force_displacement(x0, displacement_1_trial, E, n, m, a)
    F_min_req = pressure_force(x0, P_min)
    F_max_req = pressure_force(x0, P_max)

    trust_region = np.copy(stress_actual)
    trust_region[~((max_ext_max_F > -0.05 * F_max_req) & (min_ext_min_F > - 0.05 * F_min_req))] = np.nan

    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8)) = plt.subplots(4, 2, constrained_layout=True,
                                                                         sharex=False,
                                                                         sharey=False,
                                                                         figsize=(12, 12))
    plt.suptitle(f"COPS Design:\n"
                 f"L  = {x0[2] * 1000:.1f} mm, "  # \n"
                 f"w  = {x0[0] * 1000:.1f} mm, "  # \n"
                 f"t  = {x0[1] * 1000:.2f} mm, "  # \n"
                 f"d0 = {x0[3] * 1000:.1f} mm, "  # \n"
                 f"d1 = {displacement_1 * 1000:.1f} mm, "  # \n"
                 f"m  = {pss_mass(x0, args, use_cost=False) * 1000.:.1f} g",
                 weight="bold")

    norm_stress = make_color_norm(stress_compliance)

    h = ax1.contourf(w_trial * 1000., t_trial * 1000., stress_compliance, cmap="RdYlGn", norm=norm_stress)
    plt.colorbar(h, ax=ax1, label=r"$\Delta \sigma$ [Pa]")
    ax1.scatter(x0[0] * 1000, x0[1] * 1000, marker="x", color="black")
    ax1.set_ylabel(r"$t_{leg}$ [mm]")
    ax1.set_xlabel(r"$w_{leg}$ [mm]")
    ax1.set_title(r"Compliance: $\sigma_{max}$")

    k = (np.max(force_response) - np.min(force_response)) \
        / (np.max(displacement_1_trial) - np.min(displacement_1_trial)) \
        / 1000
    ax2.plot(displacement_1_trial * 1000, force_response, label=r"$F_{actual}$"rf"($k \approx {k:.1f}$ [N/mm])")
    ax2.scatter([0. * 1000], [F_min_req], marker="x", color="black")
    ax2.scatter([displacement_1_trial[-1] * 1000], [F_max_req], marker="x", color="black")
    ax2.axhline(F_min_req, color="gray", ls="dashed", label=r"$\left( F_{req}\right)_{min,max}$")
    ax2.axhline(F_max_req, color="gray", ls="dashed")
    ax2.set_xlabel(r"$\delta_{plat}$ [mm]")
    ax2.set_ylabel("Force [N]")
    ax2.set_title("Force response.")
    ax2.legend()

    norm_max_ext_max_F = make_color_norm(max_ext_max_F)
    h = ax3.contourf(w_trial * 1000., t_trial * 1000., max_ext_max_F, cmap="RdYlGn", norm=norm_max_ext_max_F)
    plt.colorbar(h, ax=ax3, label=r"$\Delta F$ [N]")
    ax3.scatter(x0[0] * 1000, x0[1] * 1000, marker="x", color="black")
    ax3.set_ylabel(r"$t_{leg}$ [mm]")
    ax3.set_title(r"Compliance: $F_{max}(\delta_{max})$")
    ax3.set_xlabel(r"$w_{leg}$ [mm]")

    norm_max_ext_min_F = make_color_norm(max_ext_min_F)
    h = ax4.contourf(w_trial * 1000., t_trial * 1000., max_ext_min_F, cmap="RdYlGn", norm=norm_max_ext_min_F)
    plt.colorbar(h, ax=ax4, label=r"$\Delta F$ [N]")
    ax4.scatter(x0[0] * 1000, x0[1] * 1000, marker="x", color="black")
    ax4.set_title(r"Compliance: $F_{min}(\delta_{max})$")
    ax4.set_ylabel(r"$t_{leg}$ [mm]")
    ax4.set_xlabel(r"$w_{leg}$ [mm]")

    norm_min_ext_max_F = make_color_norm(min_ext_max_F)
    h = ax5.contourf(w_trial * 1000., t_trial * 1000., min_ext_max_F, cmap="RdYlGn", norm=norm_min_ext_max_F)
    plt.colorbar(h, ax=ax5, label=r"$\Delta F$ [N]")
    ax5.scatter(x0[0] * 1000, x0[1] * 1000, marker="x", color="black")
    ax5.set_ylabel(r"$t_{leg}$ [mm]")
    ax5.set_xlabel(r"$w_{leg}$ [mm]")
    ax5.set_title(r"Compliance: $F_{max}(\delta_{min})$")

    norm_min_ext_min_F = make_color_norm(min_ext_min_F)
    h = ax6.contourf(w_trial * 1000., t_trial * 1000., min_ext_min_F, cmap="RdYlGn", norm=norm_min_ext_min_F)
    plt.colorbar(h, ax=ax6, label=r"$\Delta F$ [N]")
    ax6.scatter(x0[0] * 1000, x0[1] * 1000, marker="x", color="black")
    ax6.set_title(r"Compliance: $F_{min}(\delta_{min})$")
    ax6.set_ylabel(r"$t_{leg}$ [mm]")
    ax6.set_xlabel(r"$w_{leg}$ [mm]")

    norm_trust_region = make_color_norm(trust_region)
    h = ax7.contourf(w_trial * 1000., t_trial * 1000., trust_region, cmap="RdYlGn", norm=norm_trust_region)
    plt.colorbar(h, ax=ax7, label=r"$\sigma_{max}$ [Pa]")
    ax7.scatter(x0[0] * 1000, x0[1] * 1000, marker="x", color="black")
    ax7.set_ylabel(r"$t_{leg}$ [mm]")
    ax7.set_xlabel(r"$w_{leg}$ [mm]")
    ax7.set_title(r"Stress trust region ($0.95 \cdot F_{req} < F_{actual} < 1.05 \cdot F_{req}$)")

    sf_max = 5.
    sf_v_max = np.minimum(np.nanmax(safety_factor), sf_max)
    norm_sf = colors.TwoSlopeNorm(vmin=0., vcenter=1., vmax=np.maximum(sf_v_max, 1.2))
    h = ax8.contourf(w_trial * 1000., t_trial * 1000., safety_factor, cmap="RdYlGn", norm=norm_sf,
                     vmin=0., vmax=sf_v_max, levels=10)
    plt.colorbar(h, ax=ax8, label=r"SF [-]")
    ax8.scatter(x0[0] * 1000, x0[1] * 1000, marker="x", color="black")
    ax8.set_ylabel(r"$t_{leg}$ [mm]")
    ax8.set_xlabel(r"$w_{leg}$ [mm]")
    ax8.set_title(r"Safety factor ($\sigma_{max}$).")


    plt.show()


if __name__ == '__main__':
    find_opt_spring(E=110.0 * 10**9, rho=4480, sigma_max=1.103 * 10**9, SF=1.2)