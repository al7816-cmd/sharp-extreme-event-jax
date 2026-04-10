import numpy as np
import glob
import matplotlib.pyplot as plt
import re

###############################################################
# inputs

nu = "0.01"
c1 = "0.001"
c2 = "0.0"
filename = "inst_act.npy"

T="300"
filename_dns = "u.npy"

###############################################################
# constants

files = glob.glob(f"/Users/rawdata/Downloads/data/inst/nu_{nu}_c1_{c1}_c2_{c2}_obs_*_date_*_*_*_*_*_*/{filename}")
files_dns = glob.glob(f"/Users/rawdata/Downloads/data/dns/nu_{nu}_c1_{c1}_c2_{c2}_sigma_*_T_{T}_dt_*_seed_*/{filename_dns}")

###############################################################
# graphing inst files

def graph_inst_rate():
    print(f"number of files: {len(files)}")

    obs_values = []
    data_values = []

    for file in files:
        # Extract obs value from path
        match = re.search(r'obs_([\d.eE+\-]+)', file)
        if match:
            obs = float(match.group(1))
            data = np.load(file)

            # if data > 200:
            #     print(obs)

            obs_values.append(obs)
            data_values.append(data)

    sorted_pairs = sorted(zip(obs_values, data_values), key=lambda x: x[0])
    obs_values, data_values = zip(*sorted_pairs)

    # plotting the rate function for each target observation
    plt.figure()
    plt.plot(obs_values, data_values, marker='o')
    plt.xlabel(r'Target Observation $\sum u_n$')
    plt.ylabel(r'Action $I(z) = \frac{1}{2} || \eta_z || _{L^2}^2$')
    plt.title('Rate Function Plot')
    plt.figtext(0.5, 0.01, r"Figure 1: Parameters used were $\nu = 0.01$, $c_1 = 0.001$, and $c_2 = 0.0$", ha='center',
                fontsize=9)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    # plt.savefig(f"/Users/rawdata/Downloads/data/inst/rate_function.pdf")
    plt.show()

def graph_inst_decay(normalized=True):
    print(f"number of files: {len(files)}")

    obs_values = []
    data_values = []

    for file in files:
        # Extract obs value from path
        match = re.search(r'obs_([\d.eE+\-]+)', file)
        if match:
            obs = float(match.group(1))
            data = np.load(file)

            # if data > 200:
            #     print(obs)

            obs_values.append(obs)
            data_values.append(data)

    sorted_pairs = sorted(zip(obs_values, data_values), key=lambda x: x[0])
    obs_values, data_values = zip(*sorted_pairs)

    if normalized:
        y = np.exp(-np.array(data_values))
        normalization = np.trapz(y, obs_values)  # ∫ e^{-I(z)} dz
        y_normalized = y / normalization

        plt.figure()
        plt.plot(obs_values, y_normalized, marker='o')
        plt.xlabel(r'Target Observation $\sum u_n$')
        plt.ylabel(r'PDF $\rho(z) \propto e^{-I(z)}$')
        plt.title('Instanton PDF')
        plt.figtext(0.5, 0.01, r"Figure 2: Parameters used were $\nu = 0.01$, $c_1 = 0.001$, and $c_2 = 0.0$",
                    ha='center',
                    fontsize=9)
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        # plt.savefig(f"/Users/rawdata/Downloads/data/inst/exp_decay_rate.pdf")
        plt.show()
    else:
        # plotting the exponential decay rate, e^{-I(z)}
        plt.figure()
        plt.plot(obs_values, np.exp(-np.array(data_values)), marker='o')
        plt.xlabel(r'Target Observation $\sum u_n$')
        plt.ylabel(r'Exponential Decay Rate $e^{-I(z)}$')
        plt.title('Exponential Decay Rate Plot')
        plt.figtext(0.5, 0.01, r"Figure 2: Parameters used were $\nu = 0.01$, $c_1 = 0.001$, and $c_2 = 0.0$", ha='center',
                    fontsize=9)
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        # plt.savefig(f"/Users/rawdata/Downloads/data/inst/exp_decay_rate.pdf")
        plt.show()

def graph_dns():
    print(f"number of dns files: {len(files_dns)}")

    all_velo = []

    for file in files_dns:
        data = np.load(
            file)  # data := one list of length 10,000 for each file in files, each value represents the total EDR (sum over shells) at time T for a given sample
        all_velo.append(data)  # appending energy dissipation rate at time t=T=10 for all paths

    all_velo = np.concatenate(all_velo)

    print(f"number of samples: {len(all_velo)}")

    plt.style.use('default')
    fig, ax = plt.subplots()

    mean = np.mean(all_velo)
    std = np.std(all_velo)
    normalized = (all_velo - mean) / std

    hist, bins = np.histogram(normalized, bins=200, density=True)
    centers = 0.5 * (bins[:-1] + bins[1:])
    mask = hist > 0
    ax.plot(centers[mask], hist[mask], "o", markersize=3, label=r'$\nu = 10^{-2}$')

    # ax.set_yscale('log')
    ax.set_xlabel(r'$\sum u_n$')
    ax.set_ylabel(r'PDF $\rho$')
    ax.set_title('PDF of Agg Velo')
    ax.legend()
    plt.figtext(0.5, 0.01, r"Figure 2: Parameters used were $\nu = 0.01$, $c_1 = 0.001$, and $c_2 = 0.0$", ha='center',
                fontsize=9)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    # plt.savefig("f/Users/rawdata/Downloads/data/inst/pdf_agg_velo.pdf")
    plt.show()

def graph_overlay():
    # preparing DNS data
    print(f"number of dns files: {len(files_dns)}")

    all_velo = []

    for file in files_dns:
        data = np.load(
            file)  # data := one list of length 10,000 for each file in files, each value represents the total EDR (sum over shells) at time T for a given sample
        all_velo.append(data)  # appending energy dissipation rate at time t=T=10 for all paths

    all_velo = np.concatenate(all_velo)

    print(f"number of samples: {len(all_velo)}")

    plt.style.use('default')
    fig, ax = plt.subplots()

    # preparing instanton data
    print(f"number of files: {len(files)}")

    obs_values = []
    data_values = []

    for file in files:
        # Extract obs value from path
        match = re.search(r'obs_([\d.eE+\-]+)', file)
        if match:
            obs = float(match.group(1))
            data = np.load(file)

            # if data > 200:
            #     print(obs)

            obs_values.append(obs)
            data_values.append(data)

    sorted_pairs = sorted(zip(obs_values, data_values), key=lambda x: x[0])
    obs_values, data_values = zip(*sorted_pairs)

    ################### GRAPHING ##############################

    # DNS histogram: use all_velo directly, not normalized
    hist, bins = np.histogram(all_velo, bins=200, density=True)
    centers = 0.5 * (bins[:-1] + bins[1:])
    mask = hist > 0
    ax.plot(centers[mask], hist[mask], "o", markersize=3, label=r'$\nu = 10^{-2}$')

    # Instanton plot normalized to have area = 1 using the trapezoid rule
    y = np.exp(-np.array(data_values))
    normalization = np.trapz(y, obs_values)
    y_normalized = y / normalization  # no * std

    ax.plot(obs_values, y_normalized, marker="o", markersize=3, label='Instanton', color='black')

    # ax.set_yscale('log')
    ax.set_xlabel(r'$\sum u_n$')
    ax.set_ylabel(r'PDF $\rho$')
    ax.set_title('PDF of Agg Velo')
    ax.legend()
    plt.figtext(0.5, 0.01, r"Figure 2: Parameters used were $\nu = 0.01$, $c_1 = 0.001$, and $c_2 = 0.0$", ha='center',
                fontsize=9)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    # plt.savefig(f"/Users/rawdata/Downloads/data/inst/pdf_agg_velo_overlay_semilog.pdf")
    plt.show()

if __name__ == "__main__":
    # graph_dns()
    # graph_inst_decay()
    graph_overlay()