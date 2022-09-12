import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, minimize
from matplotlib.colors import BoundaryNorm
import matplotlib.colors as colors
from matplotlib import cm

def root_poly_wall(P, r_in, alpha, sigma_max):
    return None

def wall_thickness_polygon(P, r_in, sigma_max, n=6, SF=1.5):
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

    sigma_max = sigma_max / SF

    alpha = np.deg2rad(360 / (2 * n))


    t = (+ np.sqrt(9 * P ** 2 * r_in ** 2 + 12 * P * r_in ** 2 * sigma_max * np.tan(alpha) ** 2) + 3 * P * r_in) / (
                6 * sigma_max)
    # both are valid solutions, and typically both are physical.
    # However, we expect higher required wall thickness for larger pressure vessel radii -> above solution is valid
    # t = (- np.sqrt(9 * P ** 2 * r_in ** 2 + 12 * P * r_in ** 2 * sigma_max * np.tan(alpha) ** 2) + 3 * P * r_in) / (
    #         6 * sigma_max)

    return t

def cap_thickness_estimate(P, r_out, E, sigma_max, SF=5):
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
    t = np.sqrt((3 * P * r_out**2) / (4 * sigma_max))

    deflection_center = 0.171 * P * r_out**4 / (E * t**3) # approx

    for i, (dc, thick) in enumerate(zip(deflection_center, t)):
        if dc > (thick / 2):
            print(f"Cap thickness assumptions not valid for r = {r_out[i]}, t = {thick}, def_cent = {dc}, P = {P}")

    return t


def mass_hex_tank(a, t_wall, t_caps, h, rho):
    V_walls = 6 * a * t_wall * h

    V_caps = 2 * 3 * a**2 / (2 * np.tan(np.deg2rad(30))) * t_caps
    V_tot = V_walls + V_caps

    mass_tot = V_tot * rho
    mass_walls = V_walls * rho
    mass_caps = V_caps * rho

    return mass_tot, mass_walls, mass_caps


def survey_hex_tanks(P=550000, V=0.000005, margin=0.15):

    Es = [110.0 * 10**9, 189.0 * 10**9, 69.0 * 10**9, 47.6 * 10**9] # Pa
    sigmas = [1.103 * 10**9, 1.620 * 10**9, 0.480 * 10**9, 3.400 * 10**9]  # Pa
    rhos = [4480, 7850, 2770, 1840] # kg/m3
    names = ["Ti", "SS", "Al", "Epx/S-GF"]
    lss = ["solid", "dashed", "dotted", "-."]

    V_req = (1 + margin) * V
    h_min, h_max = 0.005, 0.02
    hs = np.linspace(h_min, h_max, 100)
    # regular prism
    a = np.sqrt(2 * V_req / (hs * 3 * np.sqrt(3)))

    r_in = a / (2 * np.tan(np.deg2rad(60 / 2)))
    r_out = a / (2 * np.sin(np.deg2rad(60 / 2)))

    fig, (ax1, ax3, ax2) = plt.subplots(3, 1, constrained_layout=True, sharex=True, figsize=(9, 12))

    plt.suptitle(f"HEX tank V: {V_req / 0.000001:.2f} ml ({margin * 100:.0f}% margin)")

    ax1.plot(hs * 1000, r_in * 1000, label=r"$R_{inscribed}$", c="gold")
    ax1.plot(hs * 1000, r_out * 1000, label=r"$R_{circumscribed}$", c="brown")
    ax1.set_ylabel("Radius [mm]")


    for E, sigma_max, rho, name, ls in zip(Es, sigmas, rhos, names, lss):
        t_walls = wall_thickness_polygon(P, r_in, sigma_max)
        t_caps = cap_thickness_estimate(P, r_out, E, sigma_max)

        mass_tot, mass_walls, mass_caps = mass_hex_tank(a, t_walls, t_caps, hs, rho)

        ax3.plot(hs * 1000, t_walls * 1000, label=r"$t_{w}$" + f" {name}", c="blue", ls=ls)
        ax3.plot(hs * 1000, t_caps * 1000, label=r"$t_{c}$" + f" {name}", c="red", ls=ls)
        ax3.set_ylabel("Wall thickness [mm]")

        ax2.plot(hs * 1000, mass_tot * 1000, label=r"$m_{tot}$" + f" {name}", c="green", ls=ls)
        ax2.plot(hs * 1000, mass_walls * 1000, label=r"$m_{w}$" + f" {name}", c="blue", ls=ls)
        ax2.plot(hs * 1000, mass_caps * 1000, label=r"$m_{c}$" + f" {name}", c="red", ls=ls)

    ax1.set_title(f"HEX tank geometry")
    ax1.legend()
    ax1.set_ylim(0, None)
    ax3.legend(ncol=4)
    ax3.set_ylim(0, None)

    ax2.set_title(f"HEX tank mass")
    ax2.set_xlabel("Height [mm]")
    ax2.set_ylabel("Mass [g]")
    ax2.legend(ncol=4)
    ax2.set_ylim(0, None)

    plt.show()


def get_cops_geometry(x, displacement_1, E, F, n, m, a, k_Theta=2.65, gamma=0.85):
    w, t, L, displacement_0 = x
    I = w * t ** 3 / 12
    Theta = a * np.sin(displacement_1 / (2 * gamma * L))
    Theta_0 = a * np.sin(displacement_0 / (2 * gamma * L))

    return ((n * m) / (n + m)) * ((12 * k_Theta * E * I * (Theta - Theta_0)) / (L ** 2 * np.cos(Theta))) - F

def cops_F_disp_constraint(x, displacement_1, E, F, n, m, a, k_Theta=2.65, gamma=0.85, minmax_type=1):
    """
    check if at displacement_1 the force is within the bounds specified by the pressure requirements
    :param x:
    :param displacement_1:
    :param E:
    :param F:
    :param n:
    :param m:
    :param a:
    :param k_Theta:
    :param gamma:
    :param minmax_type:
    :return:
    """

    return - minmax_type * get_cops_geometry(x, displacement_1, E, F, n, m, a, k_Theta, gamma) + minmax_type * F

def cops_max_sigma_constraint(x, displacement_1, E, sigma_max, a, k_Theta=2.65, gamma=0.85):
    w, t, L, displacement_0 = x

    c = t/2

    Theta = a * np.sin(displacement_1 / (2 * gamma * L))

    print((2 * k_Theta * E * c * (1 - gamma * (1 - np.cos(Theta))) * Theta) / (L * np.cos(Theta)) / sigma_max)

    return sigma_max - (2 * k_Theta * E * c * (1 - gamma * (1 - np.cos(Theta))) * Theta) / (L * np.cos(Theta))


def cops_mass(x, rho=np.nan):
    w, t, L, displacement_0 = x

    volume_legs = w * t * L * 2
    mass_legs = volume_legs * rho

    return mass_legs


def find_opt_spring(E, rho, sigma_max, SF, max_displacement, F_max, F_min, s=3, a=1, b=1, k_Theta=2.65, gamma=0.85):

    sigma_max = sigma_max / SF

    n = s * a
    m = s * b

    bounds = np.array([(0.2, 3.), (0.1, 2.), (2.5, 15.), (-5., -0.5)]) / 1000.  # h, t, L, displacement_0 input in mm -> converted to m
    constraints = [
        {"type": "ineq",
         "fun": cops_F_disp_constraint,
         "args": (max_displacement, E, F_min, n, m, a, k_Theta, gamma, -1)},
        {"type": "ineq",
         "fun": cops_F_disp_constraint,
         "args": (max_displacement, E, F_max, n, m, a, k_Theta, gamma, 1)},
        {"type": "ineq",
         "fun": cops_F_disp_constraint,
         "args": (0., E, F_min, n, m, a, k_Theta, gamma, -1)},
        {"type": "ineq",
         "fun": cops_F_disp_constraint,
         "args": (0, E, F_max, n, m, a, k_Theta, gamma, 1)},
        {"type": "ineq",
         "fun": cops_max_sigma_constraint,
         "args": (max_displacement, E, sigma_max, a, k_Theta, gamma)}
    ]

    n_trial = 100
    w_trial = np.linspace(0.5, 5., n_trial) / 1000.
    t_trial = np.linspace(0.2, 3., n_trial) / 1000.

    ww_trial, tt_trial = np.meshgrid(w_trial, t_trial)

    L_trial = np.full_like(ww_trial, fill_value=25. / 1000.)
    displacement_0_trial = np.full_like(ww_trial, fill_value=-0.2 / 1000.)

    x = np.array([ww_trial, tt_trial, L_trial, displacement_0_trial])

    max_ext_max_F = cops_F_disp_constraint(x, max_displacement, E, F_max, n, m, a, k_Theta, gamma, minmax_type=1)
    max_ext_min_F = cops_F_disp_constraint(x, max_displacement, E, F_min, n, m, a, k_Theta, gamma, minmax_type=-1)
    min_ext_max_F = cops_F_disp_constraint(x, 0, E, F_max, n, m, a, k_Theta, gamma, minmax_type=1)
    min_ext_min_F = cops_F_disp_constraint(x, 0., E, F_min, n, m, a, k_Theta, gamma, minmax_type=-1)
    stress = cops_max_sigma_constraint(x, max_displacement, E, sigma_max, a, k_Theta, gamma)

    trust_region = stress * ((max_ext_max_F > 0.) & (min_ext_min_F > 0.))

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, constrained_layout=True, sharex=True, figsize=(6, 12))

    norm_stress = colors.TwoSlopeNorm(vmin=np.min(stress), vcenter=0., vmax=np.max(stress))
    h = ax1.contourf(w_trial * 1000., t_trial * 1000., stress, cmap="RdYlGn", norm=norm_stress)
    plt.colorbar(h, ax=ax1, label="stress [Pa]")
    ax1.set_ylabel("leg thickness [mm]")

    norm_max_ext_max_F= colors.TwoSlopeNorm(vmin=np.min(max_ext_max_F), vcenter=0., vmax=np.max(max_ext_max_F))
    h = ax2.contourf(w_trial * 1000., t_trial * 1000., max_ext_max_F, cmap="RdYlGn", norm=norm_max_ext_max_F)
    plt.colorbar(h, ax=ax2, label="max ext force [N]")
    ax2.set_ylabel("leg thickness [mm]")

    norm_min_ext_min_F = colors.TwoSlopeNorm(vmin=np.min(min_ext_min_F), vcenter=0., vmax=np.max(min_ext_min_F))
    h = ax3.contourf(w_trial * 1000., t_trial * 1000., min_ext_min_F, cmap="RdYlGn", norm=norm_min_ext_min_F)
    plt.colorbar(h, ax=ax3, label="min ext force [N]")
    ax3.set_ylabel("leg thickness [mm]")

    h = ax4.contourf(w_trial * 1000., t_trial * 1000., trust_region, cmap="RdYlGn", norm=norm_stress)
    plt.colorbar(h, ax=ax4, label="stress [Pa]")
    ax4.set_ylabel("leg thickness [mm]")
    ax4.set_xlabel("leg width [mm]")

    plt.show()


    # result = minimize(fun=cops_mass,
    #                   x0=np.array([3., 2., 15., -1.]) / 1000,
    #                   args=rho,
    #                   method="SLSQP",
    #                   bounds=bounds,
    #                   constraints=constraints)
    #
    # print(result)

if __name__ == '__main__':
    find_opt_spring(E=110.0 * 10**9, rho=4480, sigma_max=1.103 * 10**9, SF=1.2, max_displacement=8./1000.,
                    F_max=650, F_min=125)

    # survey_hex_tanks()
