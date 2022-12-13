import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go


def get_dv(v_ex, m_sc, m_tank, m_prop):
    return v_ex * np.log((m_sc + m_tank + m_prop) / (m_sc + m_tank))


def get_tank_mass(m_prop, P=55000):
    v_tank = m_prop * 1e-3

    print(m_prop)
    t_wall = 0.5 / 1000  # mm minimum
    # assuming h/r_opt = 2 (constant surface area)

    r = (v_tank / (2 * np.pi)) ** (1. / 3)
    h = 2. * r

    rho = 4480  # kg/m3
    m_tank = 2 * np.pi * r * h * rho * t_wall + 2 * np.pi * r ** 2 * t_wall * rho

    return m_tank


def get_hex_tank_mass(m_prop, P=55000):
    v_tank = m_prop * 1e-3

    print(m_prop)
    t_wall = 0.5 / 1000  # mm minimum
    # assuming h/r_opt = 2 (constant surface area)

    r = (v_tank / (2 * np.pi)) ** (1. / 3)
    h = 2. * r

    rho = 4480  # kg/m3
    m_tank = 2 * np.pi * r * h * rho * t_wall + 2 * np.pi * r ** 2 * t_wall * rho

    return m_tank


def plot_cylinder_ratio():
    r = np.linspace(0.8, 17, 1000)
    t = 0.5 / 1000
    v = 1.

    h = (v - 2 * np.pi * r ** 2 * t) / (2 * np.pi * r * t)

    # h = (A - 2 * 2 * np.pi * r) / (np.pi * r**2)

    ratio = r / h

    # print(r)
    # print(h)
    # print(ratio)

    # ratio = np.linspace(0.1, 1000, 100)

    volume = np.pi * (r) ** 2 * h

    fig, ax = plt.subplots()

    v_rat = volume / (np.max(volume))

    full = ax.plot(ratio, v_rat, label="Constant tank mass,\nconstant wall thickness")
    ax.axvline(0.5, c="black", ls="dashed",
               label=r"$\left(V_{encl.}\right)_{max}$ at $r / h =0.5$" + "\n(analytic sol.)")
    ax.set_xlabel(r"$r / h$")
    ax.set_ylabel(r"$V_{encl.} / V_{encl., ~max}$")
    ax.set_xscale("log")

    # ax_enn = ax.twiny()
    # enn = ax_enn.plot(r, v_rat, color='blue', alpha=0.)
    # ax_enn.set_xticks(ratio, r)
    # ax_enn.xaxis.set_ticks_position('bottom')
    # ax_enn.xaxis.set_label_position('bottom')
    # ax_enn.spines['bottom'].set_position(('axes', -0.15))
    # ax_enn.spines['bottom'].set_color('blue')
    # ax_enn.tick_params(axis='x', colors='blue')
    # ax_enn.xaxis.label.set_color('blue')
    #
    # ax_knn = ax.twiny()
    # knn = ax_knn.plot(h, v_rat, color='green', alpha=0.)
    # ax_knn.set_xticks(ratio[::10], h[::10])
    # ax_knn.xaxis.set_ticks_position('bottom')
    # ax_knn.xaxis.set_label_position('bottom')
    # ax_knn.spines['bottom'].set_position(('axes', -0.3))
    # ax_knn.spines['bottom'].set_color('green')
    # ax_knn.tick_params(axis='x', colors='green')
    # ax_knn.xaxis.label.set_color('green')

    # lines = full + enn + knn
    # labels = [l.get_label() for l in lines]
    # ax.legend(lines, labels)

    ax.legend()

    plt.tight_layout()

    plt.savefig("rh_ratio_const_tank_mass.png", dpi=350, format="png")

    plt.show()


def plot_dv(v_ex=1000):
    m_sc = 0.6  # np.linspace(0.1, 10, 20)
    m_prop = np.linspace(5, 25, 200) / 1000
    m_tank = np.linspace(5, 30, 200) / 1000

    mm_tank, mm_prop = np.meshgrid(m_tank, m_prop)

    mm_actual_tank = get_tank_mass(mm_prop)
    mm_total1 = mm_tank + mm_prop
    total_mass_allowed1 = mm_total1 < 32. / 1000
    total_mass_allowed1[not np.nonzero(total_mass_allowed1)] = 0.
    total_mass_allowed1[np.nonzero(total_mass_allowed1)] = 1.

    mm_total = mm_actual_tank + mm_prop
    total_mass_allowed = mm_total < 32. / 1000
    total_mass_allowed[not np.nonzero(total_mass_allowed)] = 0.
    total_mass_allowed[np.nonzero(total_mass_allowed)] = 1.

    dv = get_dv(v_ex, m_sc, mm_tank, mm_prop)

    # print(dv.shape, mm_prop.shape, m_sc.shape, mm_tank.shape)

    fig, (ax) = plt.subplots(1, 1, figsize=1.2 * np.array([6.4, 4.8]))

    # cont = ax.contourf(mm_tank * 1000, mm_prop * 1000, mm_actual_tank * 1000, levels=25)
    cont = ax.contourf(mm_tank * 1000, mm_prop * 1000, dv * total_mass_allowed * total_mass_allowed1, levels=25,
                       alpha=1,
                       vmin=np.min(dv))
    # cont1 = ax.contourf(mm_tank * 1000, mm_prop * 1000, dv, levels=25,
    #                    alpha=0.2)
    cont2 = ax.contour(mm_tank * 1000, mm_prop * 1000, total_mass_allowed * total_mass_allowed1, levels=0,
                       linestyles="solid", colors="white")
    # cont3 = ax.contour(mm_tank * 1000, mm_prop * 1000, dv * total_mass_allowed1, levels=0,
    #                    linestyles="dashed", colors="red")
    plt.colorbar(cont, label=r"$\Delta V$ [m/s]")

    ax.text(17, 19, "approx. infeasible region\n(violates PROP-TNK-1.1)", c="white")
    ax.text(5.5, 23, "tank mass\ninfeasible", c="white")

    ax.set_ylabel("Propellant mass [g]")
    ax.set_xlabel("Tank mass [g]")

    ax.set_title(r"Cylindrical tank, $\Delta V_{max}$ ($m_{sc}=0.6$ kg, $v_{ex}=1000$ m/s)")

    plt.savefig("dV_cyl_tank_inf_approx.png", dpi=350, format="png")

    plt.show()

    # mm_sc, mm_tank, mm_prop = np.meshgrid(m_sc, m_tank, m_prop)
    # dv = get_dv(v_ex, mm_sc, mm_tank, mm_prop)
    # fig, axes = plt.subplots(3, 3)
    # axes = np.array(axes).flatten()
    # for ax in axes:

    # fig = go.Figure(data=[go.Scatter3d(
    #     x=mm_sc.flatten(),
    #     y=mm_prop.flatten(),
    #     z=mm_tank.flatten(),
    #     mode='markers',
    #     marker=dict(
    #         size=10,
    #         color=dv.flatten(),  # set color to an array/list of desired values
    #         colorscale='Viridis',  # choose a colorscale
    #         opacity=0.8
    #     )
    # )])
    #
    # fig.update_layout(scene=dict(
    #     xaxis_title='m SC',
    #     yaxis_title='m prop',
    #     zaxis_title='m tank'),
    #     width=700,
    #     margin=dict(r=20, b=10, l=10, t=10))
    #
    # # tight layout
    # fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    # fig.show()


if __name__ == '__main__':
    plot_dv()
    # plot_cylinder_ratio()
