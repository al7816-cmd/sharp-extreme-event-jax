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

###############################################################
# constants

files = glob.glob(f"/Users/rawdata/Downloads/data/inst/nu_{nu}_c1_{c1}_c2_{c2}_obs_*_date_*_*_*_*_*_*/{filename}")

if __name__ == "__main__":
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
    plt.figtext(0.5, 0.01, r"Figure 1: Parameters used were $\nu = 0.01$, $c_1 = 0.001$, and $c_2 = 0.0$", ha='center', fontsize=9)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(f"/Users/rawdata/Downloads/data/inst/rate_function.pdf")
    plt.show()


    # plotting the exponential decay rate, e^{-I(z)}
    plt.figure()
    plt.plot(obs_values, np.exp(-np.array(data_values)), marker='o')
    plt.xlabel(r'Target Observation $\sum u_n$')
    plt.ylabel(r'Exponential Decay Rate $e^{-I(z)}$')
    plt.title('Exponential Decay Rate Plot')
    plt.figtext(0.5, 0.01, r"Figure 2: Parameters used were $\nu = 0.01$, $c_1 = 0.001$, and $c_2 = 0.0$", ha='center',
                fontsize=9)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(f"/Users/rawdata/Downloads/data/inst/exp_decay_rate.pdf")
    plt.show()
