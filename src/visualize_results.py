import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from corner import corner
from pathlib import Path

from PSS_design import design_from_para

def get_data(path="MIPEGO_results_full.csv"):
    path = Path(path)

    columns = [
        "process_id",
        "Tank material",
        "COPS material",
        "N stages",
        "Leg width [mm]",
        "Leg thickness [mm]",
        "Leg length [mm]",
        "displacement_0 [mm]",
        "displacement_1 [mm]",
        "valid",
        "score",
    ]

    df = pd.read_csv(path, names=columns)

    # TODO remove this df = df.iloc[:300]

    return df

def join_csvs(path=["MIP_EGO_results_long_combined.csv", "MIP_EGO_results_long_combined_local_opt.csv"]):
    if isinstance(path, list):
        df = get_data(path[0])
        for p in path[1:]:
            df_new = get_data(p)
            df = pd.concat([df, df_new], axis=0, ignore_index=True)
    else:
        df = get_data(path)

    df.to_csv("MIP_EGO_results_long_combined_joined_local_opt.csv")


def theta2design(df):
    theta_columns = [
        "Tank material",
        "COPS material",
        "N stages",
        "Leg width [mm]",
        "Leg thickness [mm]",
        "Leg length [mm]",
        "displacement_0 [mm]",
        "displacement_1 [mm]",
    ]

    theta = df[theta_columns]

    designs = []
    delta_V = []
    cost = []
    constraint_violation = []
    for th in theta.to_numpy():
        score, x, dV, c, cv = design_from_para(th)
        designs.append(x.get())
        delta_V.append(dV.get())
        cost.append(c.get())
        constraint_violation.append(cv.get())


    df["delta V [m/s]"] = delta_V
    df["cost_penalty"] = cost
    df["constraint_violation_penalty"] = constraint_violation

    x_labels = [
        # GENERAL
        "Tank material",  # tank material    0
        "COPS material",  # cops material    1
        "N stages",  # .    N_stages         2
        "N_legs",  # .      N_legs           3
        # COPS
        "Leg width [mm]",  # .         leg width        4
        "Leg thickness [mm]",  # .     leg thickness    5
        "Leg length [mm]",  # .        leg length       6
        "displacement_0 [mm]",  # .    displacement_0   7
        "displacement_1 [mm]",  # .    displacement_1   8
        "cops_material_E",  # E                 9
        "cops_material_rho",  # rho               10
        "cops_material_sigma_yield",  # sigma_yield       11
        "cops_material_sigma_ult",  # sigma_ult         12
        # TANK
        "Tank a [mm]",  # tank a                       13
        "Tank height [mm]",  # tank height             14  # TODO: currently functionally evaluated locally, make backref
        "Tank wall thickness [mm]",  # tank wall thickness          15
        "Tank cap thickness [mm]",  # tank cap thickness           16
        "tank_material_E",  # E                            17
        "tank_material_rho",  # rho                          18
        "tank_material_sigma_yield",  # sigma_yield                  19
        "tank_material_sigma_ult",  # sigma_ult                    20
        # INTERMEDIATE RESULTS
        "cops_material_sigma_critical",  # cops_material_sigma_critical 21
        "tank_material_sigma_critical",  # tank_material_sigma_critical 22
        # CONSTRAINTS
        "cops_max_force_max_displacement_constraint",  # cops_max_force_max_displacement_constraint   23
        "cops_min_force_max_displacement_constraint",  # cops_min_force_max_displacement_constraint   24
        "cops_max_force_zero_displacement_constraint",  # cops_max_force_zero_displacement_constraint  25
        "cops_min_force_zero_displacement_constraint",  # cops_min_force_zero_displacement_constraint  26
        "cops_max_sigma_max_displacement_constraint",  # cops_max_sigma_max_displacement_constraint   27
        "cops_max_sigma_zero_displacement_constraint",  # cops_max_sigma_zero_displacement_constraint  28
        "cops_max_sigma_min_displacement_constraint",  # cops_max_sigma_min_displacement_constraint   29
        "max_mass_constraint",
        "max_volume_constraint",
        # COST
        "COPS_width_cost",  # COPS_width           30
        "COPS_thickness_cost",  # COPS_thickness       31
        "COPS_length_cost",  # COPS_length          32
        "N_stages_cost",  # N_stages             33
        # OUTPUT
        "Propellant mass [g]",
        "Tank mass [g]",
        "COPS mass [g]",
        "PSS dry mass [g]",
        "PSS volume [cm3]",
    ]

    df[x_labels] = designs

    return df


def csv2corner(path="MIPEGO_results_full.csv"):
    df = get_data(path)

    data_labels = df.columns[1:9]
    data_labels.append("score")

    m2mm = [
        "Leg width [mm]",
        "Leg thickness [mm]",
        "Leg length [mm]",
        "displacement_0 [mm]",
        "displacement_1 [mm]",
        "Tank a [mm]",
        "Tank height [mm]",
        "Tank wall thickness [mm]",
        "Tank cap thickness [mm]",
    ]
    kg2g = [
        "Propellant mass [g]",
        "Tank mass [g]",
        "COPS mass [g]",
        "PSS dry mass [g]",
    ]
    m32cm3 = ["PSS volume [cm3]"]

    df[m2mm] = df[m2mm] * 1e3
    df[kg2g] = df[kg2g] * 1e3
    df[m32cm3] = df[m32cm3] * 1e6

    score = df["score"].to_numpy()
    mean = np.mean(score)
    std = np.std(score)

    std_score = (score - mean) / std

    df["round_stds"] = np.clip(np.around(std_score), a_min=-4, a_max=0)
    data_labels.append("round_stds")

    print(df.to_string())

    # fig, (ax1) = plt.subplots(1)
    tri = sns.pairplot(df[data_labels], kind="kde", corner=True, hue="round_stds")
    # tri.map_lower(sns.kdeplot, levels=4, color=".2")

    plt.savefig("corner_plot_full.png", dpi=350, format="png")
    # plt.show()


def prep_corner_data(path="", only_valid=True, only_good=True, max_score=0.):
    if isinstance(path, list):
        df = get_data(path[0])
        for p in path[1:]:
            df_new = get_data(p)
            df = pd.concat([df, df_new], axis=0, ignore_index=True)
    else:
        df = get_data(path)

    if only_good:
        df = df.loc[lambda df: df['score'] < max_score, :]

    print(f"Preparing {len(df)} data points...")



    data_labels = df.columns[1:9].tolist()
    data_labels.append("score")

    df = theta2design(df)

    if only_valid:
        df = df.loc[lambda df: df['constraint_violation_penalty'] == 0., :]

    m2mm = [
        "Leg width [mm]",
        "Leg thickness [mm]",
        "Leg length [mm]",
        "displacement_0 [mm]",
        "displacement_1 [mm]",
        "Tank a [mm]",
        "Tank height [mm]",
        "Tank wall thickness [mm]",
        "Tank cap thickness [mm]",
    ]
    kg2g = [
        "Propellant mass [g]",
        "Tank mass [g]",
        "COPS mass [g]",
        "PSS dry mass [g]",
    ]
    m32cm3 = ["PSS volume [cm3]"]

    df[m2mm] = df[m2mm] * 1e3
    df[kg2g] = df[kg2g] * 1e3
    df[m32cm3] = df[m32cm3] * 1e6

    score = df["score"].to_numpy()
    mean = np.mean(score)
    std = np.std(score)

    std_score = (score - mean) / std

    df["round_stds"] = np.clip(np.around(std_score), a_min=-3, a_max=0)

    # this makes absolutely no sense but it doesnt work if I dont add zeros. Dont ask why I have no idea.
    df["delta V [m/s]"] = pd.to_numeric(df["delta V [m/s]"], errors='coerce') + np.zeros(len(df["delta V [m/s]"]))
    df["cost_penalty"] = pd.to_numeric(df["cost_penalty"], errors='coerce') + np.zeros(len(df["delta V [m/s]"]))
    df["constraint_violation_penalty"] = pd.to_numeric(df["constraint_violation_penalty"], errors='coerce') + np.zeros(len(df["delta V [m/s]"]))
    return df



def csv2corner_labels(df, labels, figname="corner_plot_.png", kind="kde", diag_kind="kde", hue="round_stds"):
    # df = prep_corner_data(path=path)

    if hue is None:
        df = df.drop(columns="round_stds")
        labels.remove("round_stds")

    tri = sns.pairplot(df[labels], kind=kind, diag_kind=diag_kind, corner=True, hue=hue)
    plt.savefig(figname, dpi=350, format="png")


def plot_corners(path="", hue="round_stds", max_score=0.0):
    df = prep_corner_data(path=path, max_score=max_score)

    print(f"Plotting {len(df)} data points...")

    labels_parameters = [
        "Tank material",
        "COPS material",
        "N stages",
        "Leg width [mm]",
        "Leg thickness [mm]",
        "Leg length [mm]",
        "displacement_0 [mm]",
        "displacement_1 [mm]",
        "Propellant mass [g]",
        "delta V [m/s]",
        "score",
        "round_stds",
    ]
    try:
        csv2corner_labels(df, labels_parameters, figname="corner_plot_design_parameters.png", kind="kde", diag_kind="kde", hue=hue)
    except IndexError:
        csv2corner_labels(df, labels_parameters, figname="corner_plot_design_parameters.png", kind="kde", diag_kind="kde", hue=None)


    mass_parameters = [
        "PSS volume [cm3]",
        "Tank mass [g]",
        "COPS mass [g]",
        "PSS dry mass [g]",
        "Propellant mass [g]",
        "delta V [m/s]",
        "score",
        "round_stds",
    ]
    try:
        csv2corner_labels(df, mass_parameters, figname="corner_plot_mass.png", kind="kde", diag_kind="kde", hue=hue)
    except IndexError:
        csv2corner_labels(df, mass_parameters, figname="corner_plot_mass.png", kind="kde", diag_kind="kde", hue=None)


    volume_parameters = [
        "Tank a [mm]",
        "Tank height [mm]",
        "PSS volume [cm3]",
        "Propellant mass [g]",
        "score",
        "round_stds",
    ]
    try:
        csv2corner_labels(df, volume_parameters, figname="corner_plot_volume.png", kind="kde", diag_kind="kde", hue=hue)
    except IndexError:
        csv2corner_labels(df, volume_parameters, figname="corner_plot_volume.png", kind="kde", diag_kind="kde", hue=None)


    cost_parameters = [ # commented out are all zero
        # "COPS_width_cost",
        # "COPS_thickness_cost",
        # "COPS_length_cost",
        "N stages",
        "N_stages_cost",
        "score",
        "round_stds",
    ]

    try:
        csv2corner_labels(df, cost_parameters, figname="corner_plot_cost.png", kind="kde", diag_kind="kde", hue=hue)
    except IndexError:
        csv2corner_labels(df, cost_parameters, figname="corner_plot_cost.png", kind="kde", diag_kind="kde", hue=None)


    cops_parameters = [
        "COPS material",
        "N stages",
        "Leg width [mm]",
        "Leg thickness [mm]",
        "Leg length [mm]",
        "displacement_0 [mm]",
        "displacement_1 [mm]",
        "COPS mass [g]",
        "score",
        "round_stds",
    ]
    try:
        csv2corner_labels(df, cops_parameters, figname="corner_plot_cops.png", kind="kde", diag_kind="kde", hue=hue)
    except IndexError:
        csv2corner_labels(df, cops_parameters, figname="corner_plot_cops.png", kind="kde", diag_kind="kde", hue=None)

    tank_parameters = [
        "Tank material",
        "Tank a [mm]",
        "Tank height [mm]",
        "Tank wall thickness [mm]",
        "Tank cap thickness [mm]",
        "Tank mass [g]",
        "Propellant mass [g]",
        "score",
        "round_stds",
    ]
    try:
        csv2corner_labels(df, tank_parameters, figname="corner_plot_tank.png", kind="kde", diag_kind="kde", hue=hue)
    except IndexError:
        try:
            tank_parameters = [
                "Tank material",
                "Tank a [mm]",
                "Tank height [mm]",
                "Tank cap thickness [mm]",
                "Tank mass [g]",
                "Propellant mass [g]",
                "score",
                "round_stds",
            ]
            csv2corner_labels(df, tank_parameters, figname="corner_plot_tank.png", kind="kde", diag_kind="kde", hue=hue)
        except IndexError:
            csv2corner_labels(df, tank_parameters, figname="corner_plot_tank.png", kind="kde", diag_kind="kde", hue=None)




def csv2corner_full(path="MIPEGO_results_full.csv"):
    if isinstance(path, list):
        df = get_data(path[0])
        for p in path[1:]:
            df_new = get_data(p)
            df = pd.concat([df, df_new], axis=0, ignore_index=True)
    else:
        df = get_data(path)

    df = df.loc[lambda df: df['score'] < 0., :]

    data_labels = df.columns[1:9].tolist()
    data_labels.append("score")

    df = theta2design(df)

    df = df.loc[lambda df: df['constraint_violation_penalty'] == 0., :]

    df[data_labels[3:8]] = df[data_labels[3:8]] * 1e3

    score = df["score"].to_numpy()
    mean = np.mean(score)
    std = np.std(score)

    std_score = (score - mean) / std

    df["round_stds"] = np.clip(np.around(std_score), a_min=-3, a_max=0)
    data_labels.append("round_stds")

    data_labels.append("delta V [m/s]")
    # data_labels.append("tank_a")
    # data_labels.append("tank_height")
    data_labels.append("Propellant mass [g]")
    # data_labels.append("tank_mass")
    # data_labels.append("cops_mass")
    data_labels.append("PSS dry mass [g]")
    # data_labels.append("PSS_volume")

    print(df[data_labels].to_string())
    print(len(df))

    # fig, (ax1) = plt.subplots(1)
    tri = sns.pairplot(df[data_labels], kind="kde", diag_kind="kde", corner=True, hue="round_stds")
    # tri.map_lower(sns.kdeplot, levels=4, color=".2")

    plt.savefig("corner_plot_full_design.png", dpi=350, format="png")
    # plt.show()


if __name__ == '__main__':
    path = "MIP_EGO_results_long_combined_local_opt.csv"  # ["MIP_EGO_results_full_long.csv", "MIP_EGO_results_full_long2.csv"]
    # csv2corner_full(path=path)
    plot_corners(path=path, hue="round_stds", max_score=-30.0)

