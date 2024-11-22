# Not gates Pf5 - KT

#%% Needed fuctions
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from lmfit import minimize, Parameters, report_fit
import pandas as pd
import copy
from matplotlib.ticker import FormatStrFormatter, ScalarFormatter, FixedLocator
import math

# Functions, processing and plotting the data
def read_and_extract_values(file_path, row_name):
    df = pd.read_excel(file_path)
    row_median = df[df.apply(lambda row: 'B1-A Median' in str(row), axis=1)].iloc[0]
    if (row_name == 'QacRQ1' or row_name == 'QacRQ2' or row_name == 'SrpRS1' or row_name == 'SrpRS2'
         or row_name == 'SrpRS3' or row_name == 'SrpRS4' or row_name == 'PhiFP1' or row_name == 'PhiFP2' or row_name == 'PhiFP3'):
        values_row = row_median[row_median.index.astype(str).str.contains(row_name)][3:]  # exclude 0, 5 and 10 IPTG
    else:
        values_row = row_median[row_median.index.astype(str).str.contains(row_name)][2:]  # exclude 0 and 5 IPTG
    return values_row.astype(float)

def process_set(csv_files, row_name):
    return pd.DataFrame([read_and_extract_values(file_path, row_name) for file_path in csv_files])

def organize_data_into_dataframe(sets_data, set_names):
    df_list = []

    for set_idx, set_name in zip(range(1, len(sets_data) + 1), set_names):
        for rep_idx in range(sets_data[set_idx].shape[0]):
            df_replica = pd.concat([
                sets_data[set_idx].iloc[rep_idx].reset_index(drop=True),
                pd.Series([set_name], name='Set'),
                pd.Series([rep_idx + 1], name='Replica')
            ], axis=0, ignore_index=True)
            df_list.append(df_replica)

    result_df = pd.concat(df_list, axis=1).T
    result_df.reset_index(drop=True, inplace=True)

    # Name the last two columns as "Set" and "Replica"
    result_df.columns = result_df.columns[:-2].tolist() + ['Set', 'Replica']

    # Drop rows containing NaN values
    result_df = result_df.dropna()

    return result_df

def normalize_data(df_reference, df_numerator, df_denominator):
    # Extract unique "Set" and "Replica" values from numerator DataFrame
    unique_sets_replicas = df_numerator[['Set', 'Replica']].drop_duplicates()

    normalized_dfs = []

    for _, row in unique_sets_replicas.iterrows():
        set_name = row['Set']
        replica_num = row['Replica']

        reference_row = df_reference[(df_reference['Set'] == set_name) & (df_reference['Replica'] == replica_num)].iloc[:, :-2]
        numerator_row = df_numerator[(df_numerator['Set'] == set_name) & (df_numerator['Replica'] == replica_num)].iloc[:, :-2]
        denominator_row = df_denominator[(df_denominator['Set'] == set_name) & (df_denominator['Replica'] == replica_num)].iloc[:, :-2]

        # Check if rows are found
        if reference_row.empty or numerator_row.empty or denominator_row.empty:
            print(f"Warning: Rows not found for Set {set_name}, Replica {replica_num}. Normalization skipped.")
            continue

        # Extract values for normalization
        values_reference = reference_row.values
        values_numerator = numerator_row.values
        values_denominator = denominator_row.values

        # Normalize
        if (df_numerator.equals(df_QacRQ1) or df_numerator.equals(df_QacRQ2) or df_numerator.equals(df_SrpRS1) or df_numerator.equals(df_SrpRS2)
            or df_numerator.equals(df_SrpRS3) or df_numerator.equals(df_SrpRS4) or df_numerator.equals(df_PhiFP1) or df_numerator.equals(df_PhiFP2) or df_numerator.equals(df_PhiFP3)):
            normalized_data = (values_numerator - values_reference[0][1:]) / (values_denominator[0][1:] - values_reference[0][1:])
        else:
            normalized_data = (values_numerator - values_reference) / (values_denominator - values_reference)


        # Create a DataFrame for normalized values
        columns = df_numerator.columns[:-2].tolist()
        normalized_df = pd.DataFrame(normalized_data, columns=columns)

        # Add "Set" and "Replica" columns
        normalized_df['Set'] = set_name
        normalized_df['Replica'] = replica_num

        normalized_dfs.append(normalized_df)

    return pd.concat(normalized_dfs, ignore_index=True)

def calculate_mean_and_std_errors(df):
    # Exclude the last 2 columns
    numeric_data = df.iloc[:, :-2]

    # Calculate the mean and standard errors for each column
    means = numeric_data.mean(axis=0)
    std_errors = numeric_data.sem(axis=0)

    return means, std_errors

def plot_data(df, set_name):
    fig, ax = plt.subplots(figsize=(10, 6))

    for index, row in df.iterrows():
        # Extract "Set" and "Replica" values
        set_value = row["Set"]
        replica_value = row["Replica"]

        # Exclude the last 2 columns
        data = row[:-2]

        label = f'{set_value}, Replica {replica_value}'

        ax.plot(IPTG, data, label=label, marker='o')

    ax.set_title(f'{set_name}')
    ax.set_xlabel('IPTG (µM)')
    ax.set_ylabel('Fluorescence')
    ax.legend(title='Set and Replica', loc='upper right')

    plt.show()

def plot_data_combined(df_list, set_names, title, ylabel, normalized=False):
    num_sets = len(df_list)
    num_cols = 3  # You can adjust the number of columns in the grid
    num_rows = (num_sets + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(14, 20))
    fig.suptitle(title, fontsize=16)

    for ax, df, set_name in zip(axes.flatten(), df_list, set_names):
        for index, row in df.iterrows():
            # Extract "Set" and "Replica" values
            set_value = row["Set"]
            replica_value = row["Replica"]

            # Exclude the last 2 columns
            data = row[:-2]

            label = f'{set_value}, Replica {replica_value}'
            if normalized:
                label = f'Normalized {label}'

            if(set_name == 'QacRQ1' or set_name == 'QacRQ2' or set_name == 'SrpRS1' or set_name == 'SrpRS2' or set_name == 'SrpRS3' or set_name == 'SrpRS4'
                or set_name == 'PhiFP1' or set_name == 'PhiFP2' or set_name == 'PhiFP3'):
                ax.plot(IPTG.drop(0), data, label=label, marker='o')
            else:
                ax.plot(IPTG, data, label=label, marker='o')


        ax.set_title(set_name)
        ax.set_xlabel('IPTG (µM)')
        ax.set_ylabel(ylabel)
        #ax.legend(title='Set and Replica', loc='upper right')

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust the layout to prevent overlap
    plt.show()

def plot_fits_onlyBars(mean_rows, set_names, alpha, errors=None, fit_curves=None, results=None, figure_size=(1.1, 4)):
    num_sets = len(mean_rows)

    # Determine the maximum number of parameters across all datasets    
    #max_num_params = max(len(result.params) - 2 for result in results)
    max_num_params = max(len(result) - 2 for result in results)

    # Set the number of columns in each subplot
    num_cols = 4 # max_num_params

    # Set wider width for the first column
    widths = [0.1] * 4 #max_num_params

    fig, axes = plt.subplots(num_sets, num_cols, figsize=(figure_size[0] * num_cols, figure_size[1] * num_sets), 
                             gridspec_kw={'width_ratios': widths})
    
    # Adjust the spacing between subplots
    plt.subplots_adjust(hspace=0.2, wspace=0.6)

    for i, (ax, mean_row, set_name, error, fit_curve, result) in enumerate(zip(axes, mean_rows, set_names, errors, fit_curves, results)):
        

        # Plot parameter values and errors as separate bar plots in the next columns
        param_names = [name for name in result.keys() if name not in ['Kdb', 'nb', 'a_0', 'g_0', 'beta_0', 'Kdb_0', 'n_0', 'nb_0',
                                                                       'a_1', 'g_1', 'beta_1', 'Kdb_1', 'n_1', 'nb_1',
                                                                       'a_2', 'g_2', 'beta_2', 'Kdb_2', 'n_2', 'nb_2']]

        if set_name == '1818':
            colors = plt.cm.inferno(np.linspace(0, 1, len(param_names)))  # Use viridis colormap
        else:
            colors = ['tab:purple', 'tab:olive', 'tab:grey', 'tab:cyan']  # Use viridis colormap

        for j, (param_name, color) in enumerate(zip(param_names, colors)):
            param_value = result[param_name].value
            param_error = result[param_name].stderr

            ax[j].bar([param_name], [param_value], yerr=[param_error], color=color, capsize=5, alpha=alpha,
                          label=f'{param_name} Value with Error')
           
            if (param_name == 'a'):
                ax[j].set_ylim([0, 2.2])
                ax[j].set_yticks([])
            if (param_name == 'g'):
                ax[j].set_ylim([0, 200])
                ax[j].ticklabel_format(axis='y', style='sci', scilimits=(2,2))
                ax[j].yaxis.major.formatter._useMathText = True
                ax[j].set_yticks([])
            if (param_name == 'beta'):
                ax[j].set_ylim([0, 0.42])
                ax[j].ticklabel_format(axis='y', style='sci', scilimits=(-1,-1))
                ax[j].yaxis.major.formatter._useMathText = True
                ax[j].set_xticks([0])
                ax[j].xaxis.set_major_locator(FixedLocator([0]))
                ax[j].set_xticklabels(["$\\beta$"])
                ax[j].set_yticks([])
            if (param_name == 'n'):
                if (set_name != '1818'):
                    ax[j].set_ylim([0, 3.2])
                    ax[j].set_yticks([])
                else:
                    ax[j].set_ylim([0, 3.2])
                    ax[j].set_yticks([1, 2, 3])

            if (param_name == 'gamma'):
                ax[j].ticklabel_format(axis='y', style='sci', scilimits=(-1,-1))
                ax[j].yaxis.major.formatter._useMathText = True
                ax[j].set_ylim([0, 0.9])
                ax[j].set_yticks([0.4, 0.8])
                ax[j].set_xticks([])
                ax[j].xaxis.set_major_locator(FixedLocator([0]))
                ax[j].set_xticklabels(["$\\gamma$"])
            if (param_name == 'Kd'):
                ax[j].ticklabel_format(axis='y', style='sci', scilimits=(3,3))
                ax[j].yaxis.major.formatter._useMathText = True
                ax[j].set_ylim([0, 2300])
                ax[j].set_yticks([1000, 2000])

            ax[j].tick_params(axis='both', labelsize=20)

        # If there are fewer parameters than the maximum, leave the remaining subplots blank
        for k in range(len(param_names) + 1, num_cols):
            ax[k].axis('off')

    #plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.tight_layout()
    plt.show()

def plot_fits(mean_rows, set_names, alpha, errors=None, fit_curves=None, results=None, figure_size=(1.5, 3.1)):
    num_sets = len(mean_rows)

    # Determine the maximum number of parameters across all datasets
    max_num_params = max(len(result.params) - 2 for result in results)

    # Set the number of columns in each subplot
    num_cols = 1 + max_num_params

    # Set wider width for the first column
    widths = [1] + [0.1] * max_num_params

    fig, axes = plt.subplots(num_sets, num_cols, figsize=(figure_size[0] * num_cols, figure_size[1] * num_sets), 
                             gridspec_kw={'width_ratios': widths})
    
    # Adjust the spacing between subplots
    plt.subplots_adjust(hspace=0.6, wspace=0.6)

    for i, (ax, mean_row, set_name, error, fit_curve, result) in enumerate(zip(axes, mean_rows, set_names, errors, fit_curves, results)):
        # Plot mean values with errors and fitted curves in the first column
        if error is not None:
            error = np.asarray(error).flatten()
            ax[0].errorbar(IPTG, mean_row, yerr=error, label=f'Mean {set_name}', marker='o', linestyle='None', color='#79a55b')
        else:
            ax[0].plot(IPTG, mean_row, label=f'Mean {set_name}', marker='o', linestyle='None', color='#d9798f', markersize=5)

        if fit_curve is not None:
            #ax[0].plot(np.linspace(0, 1000, 100), fit_curve, label='Fitted Curve', color='#d9798f')
            ax[0].plot(np.linspace(0, 1000, 100), fit_curve, label='Fitted Curve', color='#79a55b')

        ax[0].set_title(set_name)#, fontsize=25)
        ax[0].set_xlabel('IPTG (µM)')#, fontsize=20)
        ax[0].set_ylabel('Output (RPU)')#, fontsize=20)
        ax[0].tick_params(axis='both')#, labelsize=20)
        ax[0].set_xticks([0, 1000])
        max_y_value = myround(max(fit_curve))
        ax[0].set_yticks([max_y_value])
        ax[0].set_ylim([0, 0.8])
        #ax[0].legend()

        # Plot parameter values and errors as separate bar plots in the next columns
        param_names = [name for name in result.params.keys() if name not in ['Kdb', 'nb']]

        if set_name == '1818':
            colors = plt.cm.inferno(np.linspace(0, 1, len(param_names)))  # Use viridis colormap
        else:
            colors = ['tab:purple', 'tab:olive', 'tab:grey', 'tab:cyan']  # Use viridis colormap

        for j, (param_name, color) in enumerate(zip(param_names, colors)):
            param_value = result.params[param_name].value
            param_error = result.params[param_name].stderr

            ax[j + 1].bar([param_name], [param_value], yerr=[param_error], color=color, capsize=5, alpha=alpha,
                          label=f'{param_name} Value with Error')
           
            if (param_name == 'a'):
                ax[j + 1].set_ylim([0, 2.2])
                ax[j + 1].set_yticks([0, 1, 2])
            if (param_name == 'g'):
                ax[j + 1].set_ylim([0, 200])
                ax[j + 1].ticklabel_format(axis='y', style='sci', scilimits=(2,2))
                ax[j + 1].yaxis.major.formatter._useMathText = True
                ax[j + 1].set_yticks([0, 100, 200])
            if (param_name == 'beta'):
                ax[j + 1].set_ylim([0, 0.42])
                ax[j + 1].ticklabel_format(axis='y', style='sci', scilimits=(-1,-1))
                ax[j + 1].yaxis.major.formatter._useMathText = True
                ax[j + 1].set_xticks([0])
                ax[j + 1].xaxis.set_major_locator(FixedLocator([0]))
                ax[j + 1].set_xticklabels(["$\\beta$"])
            if (param_name == 'n'):
                ax[j + 1].set_ylim([0, 3.2])
                ax[j + 1].set_yticks([0, 1, 2, 3])
            if (param_name == 'gamma'):
                ax[j + 1].ticklabel_format(axis='y', style='sci', scilimits=(-1,-1))
                ax[j + 1].yaxis.major.formatter._useMathText = True
                ax[j + 1].set_ylim([0, 0.9])
                ax[j + 1].set_yticks([0, 0.4, 0.8])
                ax[j + 1].set_xticks([0])
                ax[j + 1].xaxis.set_major_locator(FixedLocator([0]))
                ax[j + 1].set_xticklabels(["$\\gamma$"])
            if (param_name == 'Kd'):
                ax[j + 1].ticklabel_format(axis='y', style='sci', scilimits=(3,3))
                ax[j + 1].yaxis.major.formatter._useMathText = True
                ax[j + 1].set_ylim([0, 2300])
                ax[j + 1].set_yticks([0, 1000, 2000])

            ax[j + 1].tick_params(axis='both', labelsize=20)

        # If there are fewer parameters than the maximum, leave the remaining subplots blank
        for k in range(len(param_names) + 1, num_cols):
            ax[k].axis('off')

    #plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def myround(n):
    if n == 0:
        return 0
    sgn = -1 if n < 0 else 1
    scale = int(-math.floor(math.log10(abs(n))))
    if scale <= 0:
        scale = 1
    factor = 10**scale
    return sgn*math.floor(abs(n)*factor)/factor

# Functions for Control fit (1818) and Single Gate fits
def ControlModel(x, gamma, Kd, n):
    return gamma * x**n / (Kd**n + x**n)

def ControlModel_dataset(params, x):
    gamma = params['gamma'].value
    Kd = params['Kd'].value
    n = params['n'].value
    return ControlModel(x, gamma, Kd, n)

def objective_Control(params, x, data):
    resid = data - ControlModel_dataset(params, x)
    # Check if the object is not a numpy array
    if not isinstance(resid, np.ndarray):
        resid_np = resid.to_numpy()  # Convert resid to NumPy array
    else:
        resid_np = resid
    return resid_np.flatten()

def HybridModel(x, a, g, beta, Kdb, n, nb):       
    return a * (beta + (1-beta) * 1 / (1 + (g * (x**nb / (Kdb**nb+x**nb)))**n))

def HybridModel_dataset(params, x):
    a = params['a'].value
    g = params['g'].value
    beta = params['beta'].value
    Kdb = params['Kdb'].value
    n = params['n'].value
    nb = params['nb'].value
    return HybridModel(x, a, g, beta, Kdb, n, nb)

def objective_Hybrid(params, x, data):
    resid = data - HybridModel_dataset(params, x)
    # Check if the object is not a numpy array
    if not isinstance(resid, np.ndarray):
        resid_np = resid.to_numpy()  # Convert resid to NumPy array
    else:
        resid_np = resid
    return resid_np.flatten()

# Functions for Simultaneous Fits (only changing RBS gates)
def HybridModel_simultaneousFit(x, a, g, beta, Kdb, n, nb):
    return a * (beta + (1 - beta) * 1 / (1 + (g * (x**nb / (Kdb**nb + x**nb)))**n))

def HybridModel_simultaneousFit_dataset(params, i, x):
    a = params[f'a_{i}'].value
    g = params[f'g_{i}'].value
    beta = params[f'beta_{i}'].value
    Kdb = params[f'Kdb_{i}'].value
    n = params[f'n_{i}'].value
    nb = params[f'nb_{i}'].value
    return HybridModel_simultaneousFit(x, a, g, beta, Kdb, n, nb)

def objective_Hybrid_simultaneousFit(params, x, data):
    ndata, _ = data.shape
    resid = np.zeros_like(data)
    for i in range(ndata):
        resid[i, :] = data[i, :] - HybridModel_simultaneousFit_dataset(params, i, x)
    return resid.flatten()

def perform_simultaneous_fit(IPTG, mean_values_list):
    data = np.array(mean_values_list)
    fit_params = Parameters()

    # Loop through datasets and set parameters for each dataset
    for iy in range(1, len(data) + 1):
        fit_params.add(f'a_{iy-1}', value=20, min=0)
        fit_params.add(f'g_{iy-1}', value=1.08, min=0)
        fit_params.add(f'beta_{iy-1}', value=0.2, min=0, max=0.99)
        fit_params.add(f'Kdb_{iy-1}', value=165.8665, vary=False)
        fit_params.add(f'n_{iy-1}', value=1.5, min=0.2, max=4) 
        fit_params.add(f'nb_{iy-1}', value=1.5207, vary=False)

    # Set parameters for all datasets to be equal to the first dataset
    for iy in range(2, len(data) + 1):  
        fit_params[f'a_{iy-1}'].expr = f'a_0'  
        fit_params[f'beta_{iy-1}'].expr = f'beta_0'
        fit_params[f'n_{iy-1}'].expr = f'n_0'

    # Call the function to fit datasets simultaneously
    result = minimize(
        objective_Hybrid_simultaneousFit,
        fit_params,
        args=(IPTG, data),
        method='least_squares'
    )

    return result

def plot_combined_simultaneous_fit(IPTG, host, datasets_list, std_list, names_list, results_list, alpha, figure_size=(10, 8)):
    fig, axs = plt.subplots(2, 5, figsize=figure_size, sharex=False, gridspec_kw={'width_ratios': [4.5, 0.4, 1, 0.4, 0.4]})

    x_points_fit = np.linspace(IPTG.min(), IPTG.max(), 100)
    #x_points_fit = np.linspace(0, IPTG.max(), 100)
    if "pf5" in host:
        color = '#79a55b'
    else:
        color = '#d9798f'
    
    # Adjust the spacing between subplots
    plt.subplots_adjust(hspace=1, wspace=0.6)

    for dataset, std_values, name, result, (ax_main, ax_params_a, ax_params_g, ax_params_beta, ax_params_n) in zip(datasets_list, std_list, names_list, results_list, axs):
        params = result.params

        # Plot mean values with errors for each dataset
        for i, (mean_values, std_values) in enumerate(zip(dataset, std_values)):
            print(i)
            if i==0:
                alpha_ind = 1
            elif i==1:
                alpha_ind = 0.4
            else:
                alpha_ind = 0.6


            if "SrpRS" in name and i==2:
                # Change the color to purple for SrpRS4
                ax_main.errorbar(IPTG, mean_values, yerr=std_values, label=f'Mean {name}{4}', marker='o', linestyle='None', color=color, alpha=alpha_ind)
            else:
                ax_main.errorbar(IPTG, mean_values, yerr=std_values, label=f'Mean {name}{i+1}', marker='o', linestyle='None', color=color, alpha=alpha_ind)

            # Plot simultaneous fit curve for each dataset with the same color
            fit_curve_simultaneous = HybridModel_simultaneousFit(x_points_fit, a=params[f'a_{i}'].value, g=params[f'g_{i}'].value,
                                                                 beta=params[f'beta_{i}'].value, n=params[f'n_{i}'].value,
                                                                 Kdb=params[f'Kdb_{i}'].value, nb=params[f'nb_{i}'].value)
            
            if "SrpRS" in name and i==2:
                # Change the color to purple for SrpRS4
                ax_main.plot(x_points_fit, fit_curve_simultaneous, label=f'Fit {name}{4}', linestyle='-', linewidth=2.5, color=color, alpha=alpha_ind) #color='tab:green')
            else:
                ax_main.plot(x_points_fit, fit_curve_simultaneous, label=f'Fit {name}{i+1}', linestyle='-', linewidth=2.5, color=color, alpha=alpha_ind) #color=ax_main.lines[-1].get_color())

            ax_main.set_title(name[:-1], fontsize=25)

            

        # Plot parameter 'a' as a bar plot (same for all datasets in this subplot)
        a_values = [params[f'a_{i}'].value for i in range(len(dataset))]
        a_errors = [params[f'a_{i}'].stderr for i in range(len(dataset))]
        ax_params_a.bar([f'a'], [np.mean(a_values)], yerr=[np.mean(a_errors)], color='tab:purple', alpha=alpha, capsize=5,
                      label=f'Mean a Value with Error')
        ax_params_a.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        ax_params_a.set_ylim([0, 2.2])
        ax_params_a.set_yticks([0, 1, 2])
        ax_params_a.tick_params(axis='both', labelsize=20)

        # Plot parameter 'g' as a bar plot (separate bars for each subset)
        for i in range(len(dataset)):
            g_value = params[f'g_{i}'].value
            g_error = params[f'g_{i}'].stderr
            ax_params_g.bar([f'g{i+1}'], [g_value], yerr=[g_error], capsize=5,
                          label=f'Mean g Value with Error', alpha=alpha, color='tab:olive')
        ax_params_g.yaxis.set_major_formatter(ScalarFormatter()) 
        ax_params_g.ticklabel_format(axis='y', style='sci', scilimits=(2, 2))
        ax_params_g.yaxis.major.formatter._useMathText = True
        ax_params_g.set_ylim([0, 200])
        ax_params_g.set_yticks([0, 100, 200])
        ax_params_g.tick_params(axis='both', labelsize=20)

        beta_values = params[f'beta_{i}'].value
        beta_errors = params[f'beta_{i}'].stderr
        ax_params_beta.bar([f'beta'], [beta_values], yerr=[beta_errors], color='tab:grey', alpha=alpha, capsize=5,
                        label=f'Mean beta Value with Error')
        ax_params_beta.yaxis.set_major_formatter(ScalarFormatter()) 
        ax_params_beta.set_ylim([0, 0.42])
        ax_params_beta.ticklabel_format(axis='y', style='sci', scilimits=(-1,-1))
        ax_params_beta.yaxis.major.formatter._useMathText = True
        ax_params_beta.set_xticks([0])
        ax_params_beta.set_yticks([0, 0.2, 0.4])
        ax_params_beta.xaxis.set_major_locator(FixedLocator([0]))
        ax_params_beta.set_xticklabels(["$\\beta$"])
        ax_params_beta.tick_params(axis='both', labelsize=20)

        n_values = params[f'n_{i}'].value
        n_errors = params[f'n_{i}'].stderr
        ax_params_n.bar([f'n'], [n_values], yerr=[n_errors], color='tab:cyan', alpha=alpha, capsize=5,
                        label=f'Mean n Value with Error')
        ax_params_n.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        ax_params_n.set_ylim([0, 3.2])
        ax_params_n.set_yticks([0, 1, 2, 3])
        ax_params_n.tick_params(axis='both', labelsize=20)

        # Set common labels
        ax_main.set_xlabel('IPTG (µM)')
        ax_main.set_ylabel('Output RPU)')
        ax_main.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        ax_main.set_xlabel('IPTG (µM)', fontsize=20)
        ax_main.set_ylabel('Output (RPU)', fontsize=20)
        ax_main.tick_params(axis='both', labelsize=20)
        ax_main.set_xticks([0, 1000])
        max_y_value = myround(max(fit_curve_simultaneous))
        ax_main.set_yticks([max_y_value])
        #ax_main.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=2, fontsize=12)
        #ax_main.legend(ncol=2, fontsize=15)

    # Adjust layout
    plt.tight_layout()
    plt.show()

def modify_parameters(original_params, parameters_to_rename):
    # Create a copy of the original Parameters object to avoid modifying it directly
    modified_params = copy.deepcopy(original_params)

    # Rename specified parameters
    for old_name, new_name in parameters_to_rename.items():
        if old_name in modified_params:
            modified_params[new_name] = modified_params.pop(old_name)

    return modified_params

# Function for Correction coefficient
def extract_values_corr_coef(file_path, keyword):
    # Read the Excel file
    df = pd.read_excel(file_path)

    # Find columns containing the specified keyword in the first row
    relevant_columns = [col for col in df.columns if keyword in str(col)]

    # Extract values from the fourth row for the relevant columns
    values = df.loc[2, relevant_columns].tolist()

    return values

# Function for KT reading data
def read_excel_file(file_path):
    try:
        # Read the Excel file into a DataFrame
        df = pd.read_excel(file_path, header=1)
        #df = df.iloc[2:,:] # discard 0 and 5 IPTG
        df.reset_index(inplace = True, drop = True)
        # Divide each row by the corresponding value in corr_coef
        df_divided = df.div(np.mean(corr_coef), axis=0)

        return df_divided

    except Exception as e:
        print(f"Error reading Excel file '{file_path}': {e}")
        return None

#%% Pf5 gates

################################ Read and process data #################################

# IPTG concentrations
IPTG = pd.Series([10, 20, 30, 40, 50, 70, 100, 200, 500, 1000])

# Define the paths to the Excel files for each set
base_path = 'C:/Users/pablo/Documents/CBGP/Not gates Pf5 - KT/'
sets_files = [
    ['080623 SET1', '220623 SET1 second replica', '200723 SET1 third replica'],
    ['090623 SET2', '230623 SET2 second replica', '210723 SET2 third replica'],
    ['150623 SET3', '130723 SET3 second replica', '270723 SET3 third replica'],
    ['160623 SET4', '140723 SET4 second replica', '280723 SET4 third replica']
]

# Process data for each set and store in DataFrames
sets_data_1201 = {}
sets_data_1717 = {}
sets_data_1818 = {}
sets_data_AmtrA1 = {}
sets_data_LitRL1 = {}
sets_data_AmeRF1 = {}
sets_data_HlyIIRH1 = {}
sets_data_BetIE1 = {}
sets_data_lcaRAI1 = {}
sets_data_LmrAN1 = {}
#sets_data_PsrAR1 = {}
sets_data_QacRQ1 = {}
sets_data_QacRQ2 = {}
sets_data_SrpRS1 = {}
sets_data_SrpRS2 = {}
sets_data_SrpRS3 = {}
sets_data_SrpRS4 = {}
sets_data_PhiFP1 = {}
sets_data_PhiFP2 = {}
sets_data_PhiFP3 = {}

for idx, set_dates in enumerate(sets_files, start=1):
    sets_data_1201[idx] = process_set([f'{base_path}Pf_5_NOT gates {date}.xlsx' for date in set_dates], '1201')
    sets_data_1717[idx] = process_set([f'{base_path}Pf_5_NOT gates {date}.xlsx' for date in set_dates], '1717')
    sets_data_1818[idx] = process_set([f'{base_path}Pf_5_NOT gates {date}.xlsx' for date in set_dates], '1818')
    sets_data_AmtrA1[idx] = process_set([f'{base_path}Pf_5_NOT gates {date}.xlsx' for date in set_dates], 'AmtrA1')
    sets_data_LitRL1[idx] = process_set([f'{base_path}Pf_5_NOT gates {date}.xlsx' for date in set_dates], 'LitRL1')
    sets_data_AmeRF1[idx] = process_set([f'{base_path}Pf_5_NOT gates {date}.xlsx' for date in set_dates], 'AmeRF1')
    sets_data_HlyIIRH1[idx] = process_set([f'{base_path}Pf_5_NOT gates {date}.xlsx' for date in set_dates], 'HlyIIRH1')
    sets_data_BetIE1[idx] = process_set([f'{base_path}Pf_5_NOT gates {date}.xlsx' for date in set_dates], 'BetIE1')
    sets_data_lcaRAI1[idx] = process_set([f'{base_path}Pf_5_NOT gates {date}.xlsx' for date in set_dates], 'IcaRAI1')
    sets_data_LmrAN1[idx] = process_set([f'{base_path}Pf_5_NOT gates {date}.xlsx' for date in set_dates], 'LmrAN1')
    #sets_data_PsrAR1[idx] = process_set([f'{base_path}Pf_5_NOT gates {date}.xlsx' for date in set_dates], 'PsrAR1')
    sets_data_QacRQ1[idx] = process_set([f'{base_path}Pf_5_NOT gates {date}.xlsx' for date in set_dates], 'QacRQ1')
    sets_data_QacRQ2[idx] = process_set([f'{base_path}Pf_5_NOT gates {date}.xlsx' for date in set_dates], 'QacRQ2')
    sets_data_SrpRS1[idx] = process_set([f'{base_path}Pf_5_NOT gates {date}.xlsx' for date in set_dates], 'SrpRS1')
    sets_data_SrpRS2[idx] = process_set([f'{base_path}Pf_5_NOT gates {date}.xlsx' for date in set_dates], 'SrpRS2')
    sets_data_SrpRS3[idx] = process_set([f'{base_path}Pf_5_NOT gates {date}.xlsx' for date in set_dates], 'SrpRS3')
    sets_data_SrpRS4[idx] = process_set([f'{base_path}Pf_5_NOT gates {date}.xlsx' for date in set_dates], 'SrpRS4')
    sets_data_PhiFP1[idx] = process_set([f'{base_path}Pf_5_NOT gates {date}.xlsx' for date in set_dates], 'PhiFP1')
    sets_data_PhiFP2[idx] = process_set([f'{base_path}Pf_5_NOT gates {date}.xlsx' for date in set_dates], 'PhiFP2')
    sets_data_PhiFP3[idx] = process_set([f'{base_path}Pf_5_NOT gates {date}.xlsx' for date in set_dates], 'PhiFP3')

# Organize data into a DataFrame by set and replica
set_names = ['Set1', 'Set2', 'Set3', 'Set4']
df_1201 = organize_data_into_dataframe(sets_data_1201, set_names)
df_1717 = organize_data_into_dataframe(sets_data_1717, set_names)
df_1818 = organize_data_into_dataframe(sets_data_1818, set_names)
df_AmtrA1 = organize_data_into_dataframe(sets_data_AmtrA1, set_names)
df_LitRL1 = organize_data_into_dataframe(sets_data_LitRL1, set_names)
df_AmeRF1 = organize_data_into_dataframe(sets_data_AmeRF1, set_names)
df_HlyIIRH1 = organize_data_into_dataframe(sets_data_HlyIIRH1, set_names)
df_BetIE1 = organize_data_into_dataframe(sets_data_BetIE1, set_names)
df_lcaRAI1 = organize_data_into_dataframe(sets_data_lcaRAI1, set_names)
df_LmrAN1 = organize_data_into_dataframe(sets_data_LmrAN1, set_names)
#df_PsrAR1 = organize_data_into_dataframe(sets_data_PsrAR1, set_names)
df_QacRQ1 = organize_data_into_dataframe(sets_data_QacRQ1, set_names)
df_QacRQ2 = organize_data_into_dataframe(sets_data_QacRQ2, set_names)
df_SrpRS1 = organize_data_into_dataframe(sets_data_SrpRS1, set_names)
df_SrpRS2 = organize_data_into_dataframe(sets_data_SrpRS2, set_names)
df_SrpRS3 = organize_data_into_dataframe(sets_data_SrpRS3, set_names)
df_SrpRS4 = organize_data_into_dataframe(sets_data_SrpRS4, set_names)
df_PhiFP1 = organize_data_into_dataframe(sets_data_PhiFP1, set_names)
df_PhiFP2 = organize_data_into_dataframe(sets_data_PhiFP2, set_names)
df_PhiFP3 = organize_data_into_dataframe(sets_data_PhiFP3, set_names)

# Normalizing using 1201 and 1717
normalized_df_1818 = normalize_data(df_1201, df_1818, df_1717)
normalized_df_AmtrA1 = normalize_data(df_1201, df_AmtrA1, df_1717)
normalized_df_LitRL1 = normalize_data(df_1201, df_LitRL1, df_1717)
normalized_df_AmeRF1 = normalize_data(df_1201, df_AmeRF1, df_1717)
normalized_df_HlyIIRH1 = normalize_data(df_1201, df_HlyIIRH1, df_1717)
normalized_df_BetIE1 = normalize_data(df_1201, df_BetIE1, df_1717)
normalized_df_lcaRAI1 = normalize_data(df_1201, df_lcaRAI1, df_1717)
normalized_df_LmrAN1 = normalize_data(df_1201, df_LmrAN1, df_1717)
#normalized_df_PsrAR1 = normalize_data(df_1201, df_PsrAR1, df_1717)
normalized_df_QacRQ1 = normalize_data(df_1201, df_QacRQ1, df_1717)
normalized_df_QacRQ2 = normalize_data(df_1201, df_QacRQ2, df_1717)
normalized_df_SrpRS1 = normalize_data(df_1201, df_SrpRS1, df_1717)
normalized_df_SrpRS2 = normalize_data(df_1201, df_SrpRS2, df_1717)
normalized_df_SrpRS3 = normalize_data(df_1201, df_SrpRS3, df_1717)
normalized_df_SrpRS4 = normalize_data(df_1201, df_SrpRS4, df_1717)
normalized_df_PhiFP1 = normalize_data(df_1201, df_PhiFP1, df_1717)
normalized_df_PhiFP2 = normalize_data(df_1201, df_PhiFP2, df_1717)
normalized_df_PhiFP3 = normalize_data(df_1201, df_PhiFP3, df_1717)

# Calculate means and standard errors after normalization
mean_1818, std_error_1818 = calculate_mean_and_std_errors(normalized_df_1818)
mean_AmtrA1, std_error_AmtrA1 = calculate_mean_and_std_errors(normalized_df_AmtrA1)
mean_LitRL1, std_error_LitRL1 = calculate_mean_and_std_errors(normalized_df_LitRL1)
mean_AmeRF1, std_error_AmeRF1 = calculate_mean_and_std_errors(normalized_df_AmeRF1)
mean_HlyIIRH1, std_error_HlyIIRH1 = calculate_mean_and_std_errors(normalized_df_HlyIIRH1)
mean_BetIE1, std_error_BetIE1 = calculate_mean_and_std_errors(normalized_df_BetIE1)
mean_lcaRAI1, std_error_lcaRAI1 = calculate_mean_and_std_errors(normalized_df_lcaRAI1)
mean_LmrAN1, std_error_LmrAN1 = calculate_mean_and_std_errors(normalized_df_LmrAN1)
#mean_PsrAR1, std_error_PsrAR1 = calculate_mean_and_std_errors(normalized_df_PsrAR1)
mean_QacRQ1, std_error_QacRQ1 = calculate_mean_and_std_errors(normalized_df_QacRQ1)
mean_QacRQ2, std_error_QacRQ2 = calculate_mean_and_std_errors(normalized_df_QacRQ2)
mean_SrpRS1, std_error_SrpRS1 = calculate_mean_and_std_errors(normalized_df_SrpRS1)
mean_SrpRS2, std_error_SrpRS2 = calculate_mean_and_std_errors(normalized_df_SrpRS2)
mean_SrpRS3, std_error_SrpRS3 = calculate_mean_and_std_errors(normalized_df_SrpRS3)
mean_SrpRS4, std_error_SrpRS4 = calculate_mean_and_std_errors(normalized_df_SrpRS4)
mean_PhiFP1, std_error_PhiFP1 = calculate_mean_and_std_errors(normalized_df_PhiFP1)
mean_PhiFP2, std_error_PhiFP2 = calculate_mean_and_std_errors(normalized_df_PhiFP2)
mean_PhiFP3, std_error_PhiFP3 = calculate_mean_and_std_errors(normalized_df_PhiFP3)

# Plots
plot_data(df_1201, '1201')
plot_data(df_1717, '1717')

# Combined plots
plot_data_combined([df_1818, df_AmtrA1, df_LitRL1, df_AmeRF1, df_HlyIIRH1, df_BetIE1, df_lcaRAI1, df_LmrAN1, df_QacRQ1, df_QacRQ2
                    , df_SrpRS1, df_SrpRS2, df_SrpRS3, df_SrpRS4, df_PhiFP1, df_PhiFP2, df_PhiFP3],
                   ['1818', 'AmtrA1', 'LitRL1', 'AmerF1', 'HlyIIRH1', 'BetIE1', 'lcaRAI1', 'LmrAN1', 'QacRQ1', 'QacRQ2'
                    , 'SrpRS1', 'SrpRS2', 'SrpRS3', 'SrpRS4', 'PhiFP1', 'PhiFP2', 'PhiFP3'], 'Raw Data', 'Fluorescence')
plot_data_combined([normalized_df_1818, normalized_df_AmtrA1, normalized_df_LitRL1, normalized_df_AmeRF1, normalized_df_HlyIIRH1, 
                    normalized_df_BetIE1, normalized_df_lcaRAI1, normalized_df_LmrAN1, normalized_df_QacRQ1, normalized_df_QacRQ2
                    , normalized_df_SrpRS1, normalized_df_SrpRS2, normalized_df_SrpRS3, normalized_df_SrpRS4, normalized_df_PhiFP1
                    , normalized_df_PhiFP2, normalized_df_PhiFP3],
                   ['1818', 'AmtrA1', 'LitRL1', 'AmerF1', 'HlyIIRH1', 'BetIE1', 'lcaRAI1', 'LmrAN1', 'QacRQ1', 'QacRQ2'
                    , 'SrpRS1', 'SrpRS2', 'SrpRS3', 'SrpRS4', 'PhiFP1', 'PhiFP2', 'PhiFP3'], 'Normalized Data', 'Output (RPU)', normalized=True)
        

############################## Fit Control and Single Gates ###############################

# Control 1818 parameters
fit_params_Control = Parameters()
fit_params_Control.add('gamma', value=1, min=0.01)
fit_params_Control.add('Kd', value=100, min=0.00001)
fit_params_Control.add('n', value=1, min=0.00001)

# Fit 1818
result_Control_pf5 = minimize(objective_Control, fit_params_Control, args=(IPTG, mean_1818))
params_Control_pf5 = result_Control_pf5.params

# Print results
print('Results Control pf5')
report_fit(result_Control_pf5)

# Global parameters
fit_params_Hybrid = Parameters()
fit_params_Hybrid.add('a', value=20, min=0)
fit_params_Hybrid.add('g', value=1.08, min=0)
fit_params_Hybrid.add('beta', value=0.2, min=0, max=0.99)
fit_params_Hybrid.add('Kdb', value=165.866479, vary=False)
fit_params_Hybrid.add('n', value=1.5, min=0.2, max=4)
fit_params_Hybrid.add('nb', value=1.52072006, vary=False)

# Fit gates
result_AmtrA1_pf5 = minimize(objective_Hybrid, fit_params_Hybrid, args=(IPTG, mean_AmtrA1), method='least_squares')
params_AmtrA1_pf5 = result_AmtrA1_pf5.params
result_LitRL1_pf5 = minimize(objective_Hybrid, fit_params_Hybrid, args=(IPTG, mean_LitRL1), method='least_squares')
params_LitRL1_pf5 = result_LitRL1_pf5.params
result_AmeRF1_pf5 = minimize(objective_Hybrid, fit_params_Hybrid, args=(IPTG, mean_AmeRF1), method='least_squares')
params_AmeRF1_pf5 = result_AmeRF1_pf5.params
result_HlyIIRH1_pf5 = minimize(objective_Hybrid, fit_params_Hybrid, args=(IPTG, mean_HlyIIRH1), method='least_squares')
params_HlyIIRH1_pf5 = result_HlyIIRH1_pf5.params
result_BetIE1_pf5 = minimize(objective_Hybrid, fit_params_Hybrid, args=(IPTG, mean_BetIE1), method='least_squares')
params_BetIE1_pf5 = result_BetIE1_pf5.params
result_lcaRAI1_pf5 = minimize(objective_Hybrid, fit_params_Hybrid, args=(IPTG, mean_lcaRAI1), method='least_squares')
params_lcaRAI1_pf5 = result_lcaRAI1_pf5.params
result_LmrAN1_pf5 = minimize(objective_Hybrid, fit_params_Hybrid, args=(IPTG, mean_LmrAN1), method='least_squares')
params_LmrAN1_pf5 = result_LmrAN1_pf5.params

# Calculate the corresponding y values using the fitted model
fit_curve_1818_pf5 = ControlModel_dataset(params_Control_pf5, np.linspace(0, 1000, 100))
fit_curve_AmtrA1_pf5 = HybridModel_dataset(params_AmtrA1_pf5, np.linspace(0, 1000, 100))
fit_curve_LitRL1_pf5 = HybridModel_dataset(params_LitRL1_pf5, np.linspace(0, 1000, 100))
fit_curve_AmeRF1_pf5 = HybridModel_dataset(params_AmeRF1_pf5, np.linspace(0, 1000, 100))
fit_curve_HlyIIRH1_pf5 = HybridModel_dataset(params_HlyIIRH1_pf5, np.linspace(0, 1000, 100))
fit_curve_BetIE1_pf5 = HybridModel_dataset(params_BetIE1_pf5, np.linspace(0, 1000, 100))
fit_curve_lcaRAI1_pf5 = HybridModel_dataset(params_lcaRAI1_pf5, np.linspace(0, 1000, 100))
fit_curve_LmrAN1_pf5 = HybridModel_dataset(params_LmrAN1_pf5, np.linspace(0, 1000, 100))

# Plot fits
plot_fits([mean_1818, mean_LitRL1, mean_HlyIIRH1, mean_BetIE1, mean_lcaRAI1, mean_LmrAN1],
                     ['1818', 'LitRL1', 'HlyIIRH1', 'BetIE1', 'lcaRAI1', 'LmrAN1'],
                     errors=[std_error_1818, std_error_LitRL1, std_error_HlyIIRH1, std_error_BetIE1, 
                             std_error_lcaRAI1, std_error_LmrAN1],
                     fit_curves=[fit_curve_1818_pf5, fit_curve_LitRL1_pf5, fit_curve_HlyIIRH1_pf5, fit_curve_BetIE1_pf5, 
                                 fit_curve_lcaRAI1_pf5, fit_curve_LmrAN1_pf5],
                     results=[result_Control_pf5, result_LitRL1_pf5, result_HlyIIRH1_pf5, result_BetIE1_pf5,
                               result_lcaRAI1_pf5, result_LmrAN1_pf5], alpha=0.6)












#%% Simultaneous fits

################################ Fit Simultaneous Gates #################################

# QacR
result_simultaneous_fit_QacR_pf5 = perform_simultaneous_fit(IPTG.drop(0), [mean_QacRQ1, mean_QacRQ2])
# SrpR
result_simultaneous_fit_SrpR_pf5 = perform_simultaneous_fit(IPTG.drop(0), [mean_SrpRS1, mean_SrpRS2, mean_SrpRS4])

datasets_list_pf5 = [[mean_QacRQ1, mean_QacRQ2],[mean_SrpRS1, mean_SrpRS2, mean_SrpRS4]]

std_list_pf5 = [[std_error_QacRQ1, std_error_QacRQ2],[std_error_SrpRS1, std_error_SrpRS2, std_error_SrpRS4]]

results_list_pf5 = [result_simultaneous_fit_QacR_pf5,result_simultaneous_fit_SrpR_pf5]

# Plot simultaneous fits
plot_combined_simultaneous_fit(IPTG.drop(0), 'pf5', datasets_list_pf5, std_list_pf5, ['QacRQ', 'SrpRS'], results_list_pf5, alpha=0.6)

# Parameters to rename
parameters_to_rename_0 = {'a_0': 'a','g_0': 'g','beta_0': 'beta','Kdb_0': 'Kdb','n_0': 'n','nb_0': 'nb'}
parameters_to_rename_1 = {'a_1': 'a','g_1': 'g','beta_1': 'beta','Kdb_1': 'Kdb','n_1': 'n','nb_1': 'nb'}
parameters_to_rename_2 = {'a_2': 'a','g_2': 'g','beta_2': 'beta','Kdb_2': 'Kdb','n_2': 'n','nb_2': 'nb'}

# Split the simulatenous fits into independent
result_QacRQ1_params_pf5 = modify_parameters(result_simultaneous_fit_QacR_pf5.params.copy(), parameters_to_rename_0)
result_QacRQ2_params_pf5 = modify_parameters(result_simultaneous_fit_QacR_pf5.params.copy(), parameters_to_rename_1)

result_SrpRS1_params_pf5 = modify_parameters(result_simultaneous_fit_SrpR_pf5.params.copy(), parameters_to_rename_0)
result_SrpRS2_params_pf5 = modify_parameters(result_simultaneous_fit_SrpR_pf5.params.copy(), parameters_to_rename_1)
result_SrpRS4_params_pf5 = modify_parameters(result_simultaneous_fit_SrpR_pf5.params.copy(), parameters_to_rename_2)


#%% Correction coefficient 

file_path = 'C:/Users/pablo/Documents/CBGP/Not gates Pf5 - KT/21122023KT1717vsPf51717_medians_221.xlsx'

Pf5_1201 = extract_values_corr_coef(file_path, 'Pf51201')
Pf5_1717_1 = extract_values_corr_coef(file_path, 'Pf51717 1')
Pf5_1717_2 = extract_values_corr_coef(file_path, 'Pf51717 2')
Pf5_1717_M = np.mean([Pf5_1717_1, Pf5_1717_2], axis=0)
Pf5_1717_std = np.std([Pf5_1717_1, Pf5_1717_2], axis=0)

KT2440_1201 = extract_values_corr_coef(file_path, 'KT2440 1201')
KT2440_1717_1 = extract_values_corr_coef(file_path, 'KT24401717 1')
KT2440_1717_2 = extract_values_corr_coef(file_path, 'KT24401717 2')
KT2440_1717_M = np.mean([KT2440_1717_1, KT2440_1717_2], axis=0)
KT2440_1717_std = np.std([KT2440_1717_1, KT2440_1717_2], axis=0)

corr_coef = (Pf5_1717_M - Pf5_1201) / (KT2440_1717_M - KT2440_1201)
corr_std = np.std((Pf5_1717_M - Pf5_1201) / (KT2440_1717_M - KT2440_1201))

# Plot the data points and fits with corresponding colors
fig, ax = plt.subplots(3, 2, figsize=(10, 10))

ax[0,0].set_xlabel("Input (IPTG)")
ax[0,0].set_ylabel("Output (RPU)")
ax[0,0].set_title("Pf5 - 1717 - All samples")
ax[1,0].set_xlabel("Input (IPTG)")
ax[1,0].set_ylabel("Output (RPU)")
ax[1,0].set_title("Pf5 - 1717 - Mean")
ax[2,0].set_xlabel("Input (IPTG)")
ax[2,0].set_ylabel("Output (RPU)")
ax[2,0].set_title("Pf5 - 1201")

ax[0,1].set_xlabel("Input (IPTG)")
ax[0,1].set_ylabel("Output (RPU)")
ax[0,1].set_title("KT - 1717 - All samples")
ax[1,1].set_xlabel("Input (IPTG)")
ax[1,1].set_ylabel("Output (RPU)")
ax[1,1].set_title("KT - 1717 - Mean")
ax[2,1].set_xlabel("Input (IPTG)")
ax[2,1].set_ylabel("Output (RPU)")
ax[2,1].set_title("KT - 1201")

IPTG = [0, 5, 10, 20, 30, 40, 50, 70, 100, 200, 500, 1000]

ax[0,0].errorbar(IPTG, Pf5_1717_1, yerr=np.std(Pf5_1717_1), fmt='o')
ax[0,0].errorbar(IPTG, Pf5_1717_2, yerr=np.std(Pf5_1717_2), fmt='o')
ax[1,0].errorbar(IPTG, Pf5_1717_M, yerr=Pf5_1717_std, fmt='o', color='tab:gray')

ax[0,1].errorbar(IPTG, KT2440_1717_1, yerr=np.std(KT2440_1717_1), fmt='o')
ax[0,1].errorbar(IPTG, KT2440_1717_2, yerr=np.std(KT2440_1717_2), fmt='o')
ax[1,1].errorbar(IPTG, KT2440_1717_M, yerr=KT2440_1717_std, fmt='o', color='tab:gray')

ax[2,0].errorbar(IPTG, KT2440_1201, yerr=np.std(KT2440_1201), fmt='o')
ax[2,1].errorbar(IPTG, Pf5_1201, yerr=np.std(Pf5_1201), fmt='o')

plt.tight_layout()


# Plot the data points and fits with corresponding colors
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
print(corr_std)
ax.errorbar(IPTG, corr_coef, yerr=corr_std, fmt='o')
ax.set_xlabel("Input (IPTG)")
ax.set_ylabel("Output (RPU)")
ax.set_title("Correction coefficient")









#%% KT gates

################################ Read and process data #################################

# IPTG concentrations
IPTG = pd.Series([0, 5, 10, 20, 30, 40, 50, 70, 100, 200, 500, 1000])

# EXP DATA
file_path = 'C:/Users/pablo/Documents/CBGP/Not gates Pf5 - KT/sb0c00529_si_006.xlsx'
data_frame = read_excel_file(file_path)

df_1818 = data_frame.filter(like='1818').squeeze()
df_AmtrA1 = data_frame.filter(like='AmtR_A1').squeeze()
df_LitRL1 = data_frame.filter(like='LitR_L1').squeeze()
df_AmeRF1 = data_frame.filter(like='AmeR_F1').squeeze()
df_HlyIIRH1 = data_frame.filter(like='HIYIIR_H1').squeeze()
df_BetIE1 = data_frame.filter(like='BetI_E1').squeeze()
df_lcaRAI1 = data_frame.filter(like='lcaRA_I1').squeeze()
df_LmrAN1 = data_frame.filter(like='LmrA_N1').squeeze()

df_QacRQ1 = data_frame.filter(like='QacR_Q1').squeeze()
df_QacRQ2 = data_frame.filter(like='QacR_Q2').squeeze()
df_SrpRS1 = data_frame.filter(like='SrpR_S1').squeeze()
df_SrpRS2 = data_frame.filter(like='SrpR_S2').squeeze()
df_SrpRS4 = data_frame.filter(like='SrpR_S4').squeeze()

#plt.scatter(IPTG, df_AmeRF1)
#plt.scatter(IPTG, df_1818)

############################## Fit Control and Single Gates ###############################

# Control 1818 parameters
fit_params_Control = Parameters()
fit_params_Control.add('gamma', value=0.5, min=0.00000001)
fit_params_Control.add('Kd', value=1500, min=0.0000001)
fit_params_Control.add('n', value=1, min=0.00000001)

# Fit 1818
result_Control_KT = minimize(objective_Control, fit_params_Control, args=(IPTG, df_1818))
params_Control_KT = result_Control_KT.params

# Print results
print('Results Control KT')
report_fit(result_Control_KT)

# Global parameters
fit_params_Hybrid = Parameters()
fit_params_Hybrid.add('a', value=20, min=0)
fit_params_Hybrid.add('g', value=1.08, min=0)
fit_params_Hybrid.add('beta', value=0.2, min=0, max=0.99)
fit_params_Hybrid.add('Kdb', value=1348.59131, vary=False)
fit_params_Hybrid.add('n', value=1.5, min=0.2, max=4)
fit_params_Hybrid.add('nb', value=0.90361758, vary=False)

# Fit gates
result_AmtrA1_KT = minimize(objective_Hybrid, fit_params_Hybrid, args=(IPTG, df_AmtrA1), method='least_squares')
params_AmtrA1_KT = result_AmtrA1_KT.params
result_LitRL1_KT = minimize(objective_Hybrid, fit_params_Hybrid, args=(IPTG, df_LitRL1), method='least_squares')
params_LitRL1_KT = result_LitRL1_KT.params
result_HlyIIRH1_KT = minimize(objective_Hybrid, fit_params_Hybrid, args=(IPTG, df_HlyIIRH1), method='least_squares')
params_HlyIIRH1_KT = result_HlyIIRH1_KT.params
result_BetIE1_KT = minimize(objective_Hybrid, fit_params_Hybrid, args=(IPTG, df_BetIE1), method='least_squares')
params_BetIE1_KT = result_BetIE1_KT.params
result_lcaRAI1_KT = minimize(objective_Hybrid, fit_params_Hybrid, args=(IPTG, df_lcaRAI1), method='least_squares')
params_lcaRAI1_KT = result_lcaRAI1_KT.params
result_LmrAN1_KT = minimize(objective_Hybrid, fit_params_Hybrid, args=(IPTG, df_LmrAN1), method='least_squares')
params_LmrAN1_KT = result_LmrAN1_KT.params


# Calculate the corresponding y values using the fitted model
fit_curve_1818_KT = ControlModel_dataset(params_Control_KT, np.linspace(0, 1000, 100))
fit_curve_AmtrA1_KT = HybridModel_dataset(params_AmtrA1_KT, np.linspace(0, 1000, 100))
fit_curve_LitRL1_KT = HybridModel_dataset(params_LitRL1_KT, np.linspace(0, 1000, 100))
fit_curve_HlyIIRH1_KT = HybridModel_dataset(params_HlyIIRH1_KT, np.linspace(0, 1000, 100))
fit_curve_BetIE1_KT = HybridModel_dataset(params_BetIE1_KT, np.linspace(0, 1000, 100))
fit_curve_lcaRAI1_KT = HybridModel_dataset(params_lcaRAI1_KT, np.linspace(0, 1000, 100))
fit_curve_LmrAN1_KT = HybridModel_dataset(params_LmrAN1_KT, np.linspace(0, 1000, 100))

# Plot fits (no standard errors)
plot_fits([df_1818, df_LitRL1, df_HlyIIRH1, df_BetIE1, df_lcaRAI1, df_LmrAN1],
                     ['1818', 'LitRL1', 'HlyIIRH1', 'BetIE1', 'lcaRAI1', 'LmrAN1'],
                     errors=[None, None, None, None, None, None, None],
                     fit_curves=[fit_curve_1818_KT, fit_curve_LitRL1_KT, fit_curve_HlyIIRH1_KT, fit_curve_BetIE1_KT, 
                                 fit_curve_lcaRAI1_KT, fit_curve_LmrAN1_KT],
                     results=[result_Control_KT, result_LitRL1_KT, result_HlyIIRH1_KT, result_BetIE1_KT,
                               result_lcaRAI1_KT, result_LmrAN1_KT], alpha=0.6)


# Plot fits (no standard errors)
plot_fits([df_1818, df_LitRL1], ['1818', 'LitRL1'],errors=[None, None],
                     fit_curves=[fit_curve_1818_KT, fit_curve_LitRL1_KT], results = [result_Control_KT, result_LitRL1_KT]
                     , alpha=0.6)

#%% Simultaneous fits

def perform_simultaneous_fit(IPTG, mean_values_list):
    data = np.array(mean_values_list)
    fit_params = Parameters()

    # Loop through datasets and set parameters for each dataset
    for iy in range(1, len(data) + 1):
        fit_params.add(f'a_{iy-1}', value=20, min=0)
        fit_params.add(f'g_{iy-1}', value=1.08, min=0)
        fit_params.add(f'beta_{iy-1}', value=0.2, min=0, max=0.99)
        fit_params.add(f'Kdb_{iy-1}', value=1348.59131, vary=False)
        fit_params.add(f'n_{iy-1}', value=1.5, min=0.2, max=4) 
        fit_params.add(f'nb_{iy-1}', value=0.90361758, vary=False)

    # Set parameters for all datasets to be equal to the first dataset
    for iy in range(2, len(data) + 1):  
        fit_params[f'a_{iy-1}'].expr = f'a_0'  
        fit_params[f'beta_{iy-1}'].expr = f'beta_0'
        fit_params[f'n_{iy-1}'].expr = f'n_0'

    # Call the function to fit datasets simultaneously
    result = minimize(
        objective_Hybrid_simultaneousFit,
        fit_params,
        args=(IPTG, data),
        method='least_squares'
    )

    return result

################################ Fit Simultaneous Gates #################################

# QacR
result_simultaneous_fit_QacR_KT = perform_simultaneous_fit(IPTG, [df_QacRQ1, df_QacRQ2])
# SrpR
result_simultaneous_fit_SrpR_KT = perform_simultaneous_fit(IPTG, [df_SrpRS1, df_SrpRS2, df_SrpRS4])

datasets_list_KT = [[df_QacRQ1, df_QacRQ2],[df_SrpRS1, df_SrpRS2, df_SrpRS4],]

std_list_KT = [[None, None],[None, None, None],]

results_list_KT = [result_simultaneous_fit_QacR_KT,result_simultaneous_fit_SrpR_KT]

# Plot simultaneous fits
plot_combined_simultaneous_fit(IPTG, 'KT', datasets_list_KT, std_list_KT, ['QacRQ', 'SrpRS'], results_list_KT, alpha=0.6)

# Parameters to rename
parameters_to_rename_0 = {'a_0': 'a','g_0': 'g','beta_0': 'beta','Kdb_0': 'Kdb','n_0': 'n','nb_0': 'nb'}
parameters_to_rename_1 = {'a_1': 'a','g_1': 'g','beta_1': 'beta','Kdb_1': 'Kdb','n_1': 'n','nb_1': 'nb'}
parameters_to_rename_2 = {'a_2': 'a','g_2': 'g','beta_2': 'beta','Kdb_2': 'Kdb','n_2': 'n','nb_2': 'nb'}

# Split the simulatenous fits into independent
result_QacRQ1_params_KT = modify_parameters(result_simultaneous_fit_QacR_KT.params.copy(), parameters_to_rename_0)
result_QacRQ2_params_KT = modify_parameters(result_simultaneous_fit_QacR_KT.params.copy(), parameters_to_rename_1)

result_SrpRS1_params_KT = modify_parameters(result_simultaneous_fit_SrpR_KT.params.copy(), parameters_to_rename_0)
result_SrpRS2_params_KT = modify_parameters(result_simultaneous_fit_SrpR_KT.params.copy(), parameters_to_rename_1)
result_SrpRS4_params_KT = modify_parameters(result_simultaneous_fit_SrpR_KT.params.copy(), parameters_to_rename_2)

#%% plot_params
def plot_params(params_list, names):
    # Extract values and errors for 'a', 'g', 'beta', and 'n' from all objects
    parameters = ['a', 'g', 'beta', 'n']
    colors = ['tab:purple', 'tab:olive', 'tab:grey', 'tab:cyan']

    # Create subplots
    fig, axs = plt.subplots(len(parameters), 1, figsize=(12, 3.5 * len(parameters)), sharex=True)

    for i, param in enumerate(parameters):
        for j, name in enumerate(names):
            # Extract values and errors for the current parameter from all objects
            value = params_list[j][param].value
            error = params_list[j][param].stderr

            # Set alpha based on object type
            alpha = 0.6 if "_KT" in name else 0.6

            # Adjust x-positions for all parameters
            x_position = j + ((-1) ** (j % 2)) * 0.1

            # Plot values with error bars and alpha
            if "_pf5" in name:
                axs[i].bar(x_position, value, yerr=error, color=colors[i], capsize=5, 
                            alpha=alpha, edgecolor='black', linewidth = 1.5)
            else:
                axs[i].bar(x_position, value, yerr=error, color='white', capsize=5, 
                            alpha=alpha, edgecolor='black', linewidth = 1.5)


        #axs[i].set_title(f'Parameter "{param}"')

        # Adjust x-axis ticks and labels with rotation
        x_positions_ticks = [j + ((-1) ** (j % 2)) * 0.1 for j in range(len(names))]
        axs[i].set_xticks(x_positions_ticks)
        axs[i].set_xticklabels(names, rotation=45, ha='right')
        axs[i].tick_params(axis='y', labelsize=20)
        axs[i].tick_params(axis='x', labelsize=15)

    plt.tight_layout()
    # Show the plot
    plt.show()

# Example usage
params_list = [params_LitRL1_pf5, params_LitRL1_KT,
               params_HlyIIRH1_pf5, params_HlyIIRH1_KT, params_BetIE1_pf5, params_BetIE1_KT,
               params_lcaRAI1_pf5, params_lcaRAI1_KT, params_LmrAN1_pf5, params_LmrAN1_KT, 
               result_QacRQ1_params_pf5, result_QacRQ1_params_KT, result_QacRQ2_params_pf5, 
               result_QacRQ2_params_KT, result_SrpRS1_params_pf5, result_SrpRS1_params_KT, 
               result_SrpRS2_params_pf5, result_SrpRS2_params_KT, result_SrpRS4_params_pf5,
               result_SrpRS4_params_KT]

object_names = ['LitRL1_pf5', 'LitRL1_KT',
                'HlyIIRH1_pf5', 'HlyIIRH1_KT', 'BetIE1_pf5', 'BetIE1_KT',
                'lcaRAI1_pf5', 'lcaRAI1_KT', 'LmrAN1_pf5', 'LmrAN1_KT', 
                'QacRQ1_pf5', 'QacRQ1_KT', 'QacRQ2_pf5', 'QacRQ2_KT',
                'SrpRS1_pf5', 'SrpRS1_KT', 'SrpRS2_pf5', 'SrpRS2_KT', 
                'SrpRS4_pf5', 'SrpRS4_KT']

plot_params(params_list, object_names)

#%% C0/K0 ratios

def plot_ratios(params_pf5, params_KT, gamma_controls, object_names):
    # pf5
    ratio_value_pf5 = []
    ratio_std_pf5 = []
    for param in params_pf5:
        ratio_value = param["a"].value / gamma_controls[0].value
        ratio_std = ratio_value * np.sqrt((param["a"].stderr / param["a"].value)**2 + 
                                           (gamma_controls[0].stderr / gamma_controls[0].value)**2)
        ratio_value_pf5.append(ratio_value)
        ratio_std_pf5.append(ratio_std)

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.bar(np.arange(len(ratio_value_pf5)), ratio_value_pf5, yerr=ratio_std_pf5, 
           capsize=5, color='#79a55b', edgecolor='black', linewidth=1.5)
    
    # Set x-axis tick labels
    ax.set_xticks(np.arange(len(object_names)))
    ax.set_xticklabels(object_names, rotation=45, ha='right')  # Adjust rotation and alignment as needed
    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=15)

    # Show the plot
    plt.show()

    # KT
    ratio_value_KT = []
    ratio_std_KT = []
    for param in params_KT:
        ratio_value = param["a"].value / gamma_controls[1].value
        ratio_std = ratio_value * np.sqrt((param["a"].stderr / param["a"].value)**2 + 
                                           (gamma_controls[1].stderr / gamma_controls[1].value)**2)
        ratio_value_KT.append(ratio_value)
        ratio_std_KT.append(ratio_std)

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.bar(np.arange(len(ratio_value_KT)), ratio_value_KT, yerr=ratio_std_KT, 
           capsize=5, color='#d9798f', edgecolor='black', linewidth=1.5)
    
    # Set x-axis tick labels
    ax.set_xticks(np.arange(len(object_names)))
    ax.set_xticklabels(object_names, rotation=45, ha='right')  # Adjust rotation and alignment as needed
    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=15)

    # Show the plot
    plt.show()
    

# Example usage
params_pf5 = [params_LitRL1_pf5, params_HlyIIRH1_pf5, params_BetIE1_pf5,
               params_lcaRAI1_pf5, params_LmrAN1_pf5, result_QacRQ1_params_pf5, 
               result_QacRQ2_params_pf5, result_SrpRS1_params_pf5, 
               result_SrpRS2_params_pf5, result_SrpRS4_params_pf5]

params_KT = [params_LitRL1_KT, params_HlyIIRH1_KT, params_BetIE1_KT, 
                  params_lcaRAI1_KT, params_LmrAN1_KT, result_QacRQ1_params_KT, 
                  result_QacRQ2_params_KT, result_SrpRS1_params_KT, 
                  result_SrpRS2_params_KT, result_SrpRS4_params_KT]

gamma_controls = [result_Control_pf5.params["gamma"], result_Control_KT.params["gamma"]]

object_names = ['LitRL1', 'HlyIIRH1', 'BetIE1', 'lcaRAI1', 'LmrAN1', 
                'QacRQ1', 'QacRQ2', 'SrpRS1', 'SrpRS2', 'SrpRS4']

plot_ratios(params_pf5, params_KT, gamma_controls, object_names)


#%% plot_params_ratios
def plot_params_ratios(params_list, names, gamma_controls):
    # Extract values and errors for 'a', 'g', 'beta', and 'n' from all objects
    parameters = ['a']
    colors = ['tab:orange']

    # Create subplots
    fig, axs = plt.subplots(len(parameters), 1, figsize=(12, 4.5 * len(parameters)), sharex=True)

    for i, param in enumerate(parameters):
        for j, name in enumerate(names):
            # Extract values and errors for the current parameter from all objects
            value = params_list[j][param].value
            error = params_list[j][param].stderr

            # ratio
            ratio_value_pf5 = value / gamma_controls[0].value
            ratio_std_pf5 = ratio_value_pf5 * np.sqrt((error / value)**2 + (gamma_controls[0].stderr / gamma_controls[0].value)**2)

            ratio_value_KT = value / gamma_controls[1].value
            ratio_std_KT = ratio_value_KT * np.sqrt((error / value)**2 + (gamma_controls[0].stderr / gamma_controls[0].value)**2)

            # Set alpha based on object type
            alpha = 0.6 if "_KT" in name else 0.6

            # Adjust x-positions for all parameters
            x_position = j + ((-1) ** (j % 2)) * 0.1

            # Plot values with error bars and alpha
            if "_pf5" in name:
                axs.bar(x_position, ratio_value_pf5, yerr=ratio_std_pf5, color=colors[i], capsize=5, 
                            alpha=alpha, edgecolor='black', linewidth = 1.5)
            else:
                axs.bar(x_position, ratio_value_KT, yerr=ratio_std_KT, color='white', capsize=5, 
                            alpha=alpha, edgecolor='black', linewidth = 1.5)


        #axs[i].set_title(f'Parameter "{param}"')

        # Adjust x-axis ticks and labels with rotation
        x_positions_ticks = [j + ((-1) ** (j % 2)) * 0.1 for j in range(len(names))]
        axs.set_xticks(x_positions_ticks)
        axs.set_xticklabels(names, rotation=45, ha='right')
        axs.tick_params(axis='y', labelsize=20)
        axs.tick_params(axis='x', labelsize=15)

    plt.tight_layout()
    # Show the plot
    plt.show()

# Example usage
params_list = [params_LitRL1_pf5, params_LitRL1_KT,
               params_HlyIIRH1_pf5, params_HlyIIRH1_KT, params_BetIE1_pf5, params_BetIE1_KT,
               params_lcaRAI1_pf5, params_lcaRAI1_KT, params_LmrAN1_pf5, params_LmrAN1_KT, 
               result_QacRQ1_params_pf5, result_QacRQ1_params_KT, result_QacRQ2_params_pf5, 
               result_QacRQ2_params_KT, result_SrpRS1_params_pf5, result_SrpRS1_params_KT, 
               result_SrpRS2_params_pf5, result_SrpRS2_params_KT, result_SrpRS4_params_pf5,
               result_SrpRS4_params_KT]

object_names = ['LitRL1_pf5', 'LitRL1_KT',
                'HlyIIRH1_pf5', 'HlyIIRH1_KT', 'BetIE1_pf5', 'BetIE1_KT',
                'lcaRAI1_pf5', 'lcaRAI1_KT', 'LmrAN1_pf5', 'LmrAN1_KT', 
                'QacRQ1_pf5', 'QacRQ1_KT', 'QacRQ2_pf5', 'QacRQ2_KT',
                'SrpRS1_pf5', 'SrpRS1_KT', 'SrpRS2_pf5', 'SrpRS2_KT', 
                'SrpRS4_pf5', 'SrpRS4_KT']

gamma_controls = [result_Control_pf5.params["gamma"], result_Control_KT.params["gamma"]]

plot_params_ratios(params_list, object_names, gamma_controls)


#%% Scatter params
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.stats import linregress
from scipy.optimize import curve_fit

def exponential_func(x, a, b, c):
    return a * np.exp(b * x) + c

def scatter_parameters_pandas(params_list, names, parameter_name, ax=None):
    # Create a DataFrame from the parameter values
    data = {'Name': [name.split('_')[0] for name in names],
            'pf5': [params[parameter_name].value if 'pf5' in name else None for params, name in zip(params_list, names)],
            'KT': [params[parameter_name].value if 'KT' in name else None for params, name in zip(params_list, names)]}
    df = pd.DataFrame(data)

    # Group by 'Name' and aggregate the values, keeping only non-null values
    df_cleaned = df.groupby('Name').agg({'pf5': 'first', 'KT': 'first'}).reset_index()

    # Drop rows with missing values
    df_cleaned = df_cleaned.dropna(subset=['pf5', 'KT'])

    # Set up scatter plot with seaborn
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
        
    if (parameter_name == 'a'):
        ax.set_ylim([-0.05, 2])
        ax.set_xlim([-0.1, 2.2])
    if (parameter_name == 'g'):
        ax.set_ylim([-50, 200])
        ax.set_xlim([-5, 45])
    if (parameter_name == 'beta'):
        ax.set_ylim([-0.10, 0.20])
        ax.set_xlim([-0.03, 0.15])
    if (parameter_name == 'n'):
        ax.set_ylim([0.85, 1.9])
        ax.set_xlim([0, 2.6])

    # Get the color for the parameter
    colors = ['tab:purple', 'tab:olive', 'tab:grey', 'tab:cyan']
    color = colors[parameters_to_compare.index(parameter_name)]

    sns.scatterplot(data=df_cleaned, x='KT', y='pf5', color=color, alpha=0.7, s=100, ax=ax)

    # Add labels and title
    ax.set_xlabel(f'{parameter_name} (KT)')#, fontsize=20)
    ax.set_ylabel(f'{parameter_name} (pf5)')#, fontsize=20)
    ax.tick_params(axis='y')#, labelsize=15)
    ax.tick_params(axis='x')#, labelsize=15)
    #ax.set_title(f'Comparison of Parameter {parameter_name} between pf5 and KT')

    # Calculate and plot regression line using seaborn's regplot for all parameters except 'g'
    if parameter_name != 'g':
        slope, intercept, r_value, p_value, std_err = linregress(df_cleaned['KT'], df_cleaned['pf5'])
        line_equation = f'y = {slope:.2f}x + {intercept:.2f}'

        # Extend the range of x-values
        x_range = np.linspace(df_cleaned['pf5'].min()-50, df_cleaned['pf5'].max()+100, 100)

        # Plot the regression line with extended x-range
        ax.plot(x_range, slope * x_range + intercept, color=color, linestyle='--', label=f'Regression Line: {line_equation}')

        # Calculate confidence interval based on actual data
        predictions = slope * df_cleaned['KT'] + intercept
        residuals = df_cleaned['pf5'] - predictions
        s_residuals = np.std(residuals)
        conf_int = 1.96 * s_residuals  # 95% confidence interval

        # Plot shaded area for confidence interval
        ax.fill_between(x_range, slope * x_range + intercept - conf_int, slope * x_range + intercept + conf_int, color=color, alpha=0.15, label='95% Confidence Interval')

        # Add annotations for each point
        for i, row in df_cleaned.iterrows():
            ax.annotate(row['Name'], (row['KT'], row['pf5']), fontsize=8, ha='right')

    # Fit exponential function for parameter 'g' and plot it
    if parameter_name == 'g':
        # Provide an initial guess for the parameters
        initial_guess = [1.0, 0.5, 0]  # Initial guess for parameters a, b, c
        
        # Perform curve fitting with initial guess
        popt, pcov = curve_fit(exponential_func, df_cleaned['KT'], df_cleaned['pf5'], p0=initial_guess)
        x_range = np.linspace(-5, 48, 100)
        ax.plot(x_range, exponential_func(x_range, *popt), color=color, linestyle='--', label=f'Exponential Fit')
        
        # Calculate confidence interval based on actual data
        predictions = exponential_func(df_cleaned['KT'], *popt)
        residuals = df_cleaned['pf5'] - predictions
        s_residuals = np.std(residuals)
        conf_int = 1.96 * s_residuals  # 95% confidence interval
        
        # Plot shaded area for confidence interval
        ax.fill_between(x_range, exponential_func(x_range, *popt) - conf_int, exponential_func(x_range, *popt) + conf_int, color=color, alpha=0.15, label='95% Confidence Interval')
        
        # Add annotations for each point
        for i, row in df_cleaned.iterrows():
            ax.annotate(row['Name'], (row['KT'], row['pf5']), fontsize=8, ha='right')
        


    # Calculate the Pearson correlation coefficient
    correlation_coefficient = df_cleaned['KT'].corr(df_cleaned['pf5'])

    # Add the correlation value and equation to the legend
    #legend_text = f'Pearson Correlation: {correlation_coefficient:.2f}'
    #ax.legend(title=legend_text, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=5)


params_list = [params_LitRL1_pf5, params_LitRL1_KT,
               params_HlyIIRH1_pf5, params_HlyIIRH1_KT, params_BetIE1_pf5, params_BetIE1_KT,
               params_lcaRAI1_pf5, params_lcaRAI1_KT, params_LmrAN1_pf5, params_LmrAN1_KT, 
               result_QacRQ1_params_pf5, result_QacRQ1_params_KT, result_QacRQ2_params_pf5, 
               result_QacRQ2_params_KT, result_SrpRS1_params_pf5, result_SrpRS1_params_KT, 
               result_SrpRS2_params_pf5, result_SrpRS2_params_KT, result_SrpRS4_params_pf5,
               result_SrpRS4_params_KT] 

object_names = ['LitRL1_pf5', 'LitRL1_KT',
                'HlyIIRH1_pf5', 'HlyIIRH1_KT', 'BetIE1_pf5', 'BetIE1_KT',
                'lcaRAI1_pf5', 'lcaRAI1_KT', 'LmrAN1_pf5', 'LmrAN1_KT', 
                'QacRQ1_pf5', 'QacRQ1_KT', 'QacRQ2_pf5', 'QacRQ2_KT',
                'SrpRS1_pf5', 'SrpRS1_KT', 'SrpRS2_pf5', 'SrpRS2_KT', 
                'SrpRS4_pf5', 'SrpRS4_KT'] 

parameters_to_compare = ['a', 'g', 'beta', 'n']
fig, axes = plt.subplots(2, 2, figsize=(11, 8))

for i, parameter_name in enumerate(parameters_to_compare):
    scatter_parameters_pandas(params_list, object_names, parameter_name, ax=axes[i // 2, i % 2])

plt.tight_layout()
plt.show()



#%% Scatter params - Leave-One-Out Cross-Validation 

def scatter_parameters_pandas_with_exclusion(params_list, names, parameter_name, ax=None, exclude_object=None):
    # Create a DataFrame from the parameter values
    data = {'Name': [name.split('_')[0] for name in names],
            'pf5': [params[parameter_name].value if 'pf5' in name else None for params, name in zip(params_list, names)],
            'KT': [params[parameter_name].value if 'KT' in name else None for params, name in zip(params_list, names)]}
    df = pd.DataFrame(data)

    # Group by 'Name' and aggregate the values, keeping only non-null values
    df_cleaned = df.groupby('Name').agg({'pf5': 'first', 'KT': 'first'}).reset_index()

    # Drop rows with missing values
    df_cleaned = df_cleaned.dropna(subset=['pf5', 'KT'])

    # Set up scatter plot with seaborn
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4)) 

    if (parameter_name == 'a'):
        ax.set_ylim([-0.05, 2])
        ax.set_xlim([-0.1, 2.2])
    if (parameter_name == 'g'):
        ax.set_ylim([-50, 200])
        ax.set_xlim([-5, 45])
    if (parameter_name == 'beta'):
        ax.set_ylim([-0.10, 0.20])
        ax.set_xlim([-0.03, 0.15])
    if (parameter_name == 'n'):
        ax.set_ylim([0.85, 1.9])
        ax.set_xlim([0, 2.6])

    # Get the color for the parameter
    colors = ['tab:purple', 'tab:olive', 'tab:grey', 'tab:cyan']
    color = colors[parameters_to_compare.index(parameter_name)]

    # Add labels and title
    ax.set_xticks([])
    ax.set_yticks([])
    """ ax.set_xlabel(f'{parameter_name} (KT2440)', fontsize=20)
    ax.set_ylabel(f'{parameter_name} (Pf-5)', fontsize=20)
    ax.tick_params(axis='y', labelsize=15)
    ax.tick_params(axis='x', labelsize=15) """

    # Filter rows where 'Name' is not in the list of excluded objects
    df_final_cleaned = df_cleaned[~df_cleaned['Name'].isin(exclude_object)]

    # Calculate and plot regression line using seaborn's regplot
    if parameter_name != 'g':
        slope, intercept, r_value, p_value, std_err = linregress(df_final_cleaned['KT'], df_final_cleaned['pf5'])
        line_equation = f'y = {slope:.2f}x + {intercept:.2f}'
    
        # Retain slope and intercept values for the current exclusion
        globals()[f'slope_{parameter_name}_exclude_{exclude_object}'] = slope
        globals()[f'intercept_{parameter_name}_exclude_{exclude_object}'] = intercept

        # Scatter plot for the final cleaned data
        sns.scatterplot(data=df_final_cleaned, x='KT', y='pf5', color=color, alpha=0.7, s=100, ax=ax)#, label=f'{parameter_name}')

        # Extend the range of x-values
        x_range = np.linspace(df_final_cleaned['pf5'].min()-50, df_final_cleaned['pf5'].max()+100, 100)

        # Plot the regression line with extended x-range
        ax.plot(x_range, slope * x_range + intercept, color=color, linestyle='--')#, label=f'Regression Line: {line_equation}')

        # Calculate confidence interval based on actual data
        predictions = slope * df_final_cleaned['KT'] + intercept
        residuals = df_final_cleaned['pf5'] - predictions
        s_residuals = np.std(residuals)
        conf_int = 1.96 * s_residuals  # 95% confidence interval
        #conf_int = 2.576 * s_residuals  # 99% confidence interval

        # Plot shaded area for confidence interval
        ax.fill_between(x_range, slope * x_range + intercept - conf_int, slope * x_range + intercept + conf_int, color=color, alpha=0.15)#, label='95% Confidence Interval')
    else:  # For 'g' parameter
        # Scatter plot for the final cleaned data
        sns.scatterplot(data=df_final_cleaned, x='KT', y='pf5', color=color, alpha=0.7, s=100, ax=ax)#, label=f'{parameter_name}')
        
        # Provide an initial guess for the parameters
        initial_guess = [1.0, 0.5, 0]  # Initial guess for parameters a, b, c
        
        # Perform curve fitting with initial guess
        popt, pcov = curve_fit(exponential_func, df_final_cleaned['KT'], df_final_cleaned['pf5'], p0=initial_guess)
        x_range = np.linspace(-5, 48, 100)

        # Plot the exponential fit
        ax.plot(x_range, exponential_func(x_range, *popt), color=color, linestyle='--')#, label=f'Exponential Fit')

        # Calculate confidence interval based on actual data
        predictions = exponential_func(df_final_cleaned['KT'], *popt)
        residuals = df_final_cleaned['pf5'] - predictions
        s_residuals = np.std(residuals)
        conf_int = 1.96 * s_residuals  # 95% confidence interval

        # Plot shaded area for confidence interval
        ax.fill_between(x_range, exponential_func(x_range, *popt) - conf_int, exponential_func(x_range, *popt) + conf_int, color=color, alpha=0.15)#, label='95% Confidence Interval')

    # Calculate the Pearson correlation coefficient for cleaned data
    correlation_coefficient_cleaned = df_final_cleaned['KT'].corr(df_final_cleaned['pf5'])

        # '#79a55b'
    if exclude_object:
        # Create a DataFrame for excluded data
        df_excluded = df[df['Name'].str.contains('|'.join(exclude_object))]

        # Check if the DataFrame is not empty before proceeding
        if not df_excluded.empty:
            df_excluded_combined = df_excluded.iloc[0].combine_first(df_excluded.iloc[1])
            excluded_pf5 = df_excluded_combined['pf5']
            excluded_KT = df_excluded_combined['KT']

            # Create a DataFrame for 'Excluded'
            df_excluded_single = pd.DataFrame({'Name': ['Excluded'], 'pf5': [excluded_pf5], 'KT': [excluded_KT]})
            # Plot 'Excluded' separately in orange using sns.scatterplot
            sns.scatterplot(data=df_excluded_single, x='KT', y='pf5', color='white', edgecolor='black', s=100, ax=ax)#, label=f'Excluded {parameter_name}')
            
            # Add annotations for each point
            #for i, row in df_excluded_single.iterrows():
                #ax.annotate(df_excluded_combined['Name'], (row['KT'], row['pf5']), fontsize=20, ha='right')

    # Add the correlation value and equation to the legend
    #legend_text = f'Pearson Correlation (Excluding {", ".join(exclude_object)}): {correlation_coefficient_cleaned:.2f}'
    #ax.legend(title=legend_text, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=5)

    # Return slope and intercept for later use
    if parameter_name != 'g':
        return slope, intercept
    else:
        return popt  # Return the exponential parameters b and c



# Example usage with systematic exclusion
parameters_to_compare = ['a', 'g', 'beta', 'n']
exclude_objects = ['LitRL1', 'HlyIIRH1', 'BetIE1', 'lcaRAI1', 'LmrAN1', 'QacRQ1', 'QacRQ2', 'SrpRS1', 'SrpRS2', 'SrpRS4']  # Add the names of objects to exclude as strings

# Create dictionaries to store slopes and intercepts for each excluded object
all_slopes = {exclude_object: [None] * len(parameters_to_compare) for exclude_object in exclude_objects}
all_intercepts = {exclude_object: [None] * len(parameters_to_compare) for exclude_object in exclude_objects}
all_popts = {exclude_object: [None] * len(parameters_to_compare) for exclude_object in exclude_objects}

for exclude_object in exclude_objects:
    # Convert exclude_object from string to list
    exclude_object_list = [exclude_object]
    fig, axes = plt.subplots(2, 2, figsize=(9, 6))

    slope = [None] * len(parameters_to_compare)
    intercept = [None] * len(parameters_to_compare)

    popt = [None] * len(parameters_to_compare)

    for i, parameter_name in enumerate(parameters_to_compare):
        if parameter_name != 'g':
            slope[i], intercept[i] = scatter_parameters_pandas_with_exclusion(params_list, object_names, parameter_name, ax=axes[i // 2, i % 2], exclude_object=exclude_object_list)
        else:
            popt[i] = scatter_parameters_pandas_with_exclusion(params_list, object_names, parameter_name, ax=axes[i // 2, i % 2], exclude_object=exclude_object_list)

        # Add a title to the subplot indicating the excluded object
        #axes[i // 2, i % 2].set_title(f'Excluded Gate: {exclude_object_list}')

    # Store slopes and intercepts in the dictionaries
    all_slopes[exclude_object] = slope
    all_intercepts[exclude_object] = intercept
    popt = popt[1]
    all_popts[exclude_object] = popt

    plt.tight_layout()
    plt.show()




#%% Prediction
def plot_fits_prediction(set_names, data=None, errors=None, fit_curves=None, pred_curves=None, figure_size=(1.27, 2.7)):
    num_sets = len(set_names)

    # Calculate the number of rows and columns for the subplots
    num_rows = num_sets // 2 + num_sets % 2  # Make sure to add an extra row for odd number of sets
    num_cols = 2

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(figure_size[0] * 6, figure_size[1] * num_rows))
    #fig.suptitle('Prediction', fontsize=16)

    # Flatten the axes array if there is more than one row
    axes = axes.flatten()

    for i, (ax, set_name, datum, error, fit_curve, pred_curve) in enumerate(zip(axes, set_names, data, errors, fit_curves, pred_curves)):
        if fit_curve is not None:
            if len(datum)==9:
                IPTG = pd.Series([20, 30, 40, 50, 70, 100, 200, 500, 1000])
            else:
                IPTG = pd.Series([10, 20, 30, 40, 50, 70, 100, 200, 500, 1000])
            #ax.scatter(IPTG, datum, marker='o', linestyle='None', color='#79a55b',zorder=7)
            #ax.scatter(IPTG, datum, marker='o', linestyle='None', color='white', edgecolor='black',zorder=7)
            ax.errorbar(IPTG, datum, yerr=error, marker='o', linestyle='None', color='#79a55b',zorder=5)
            #ax.plot(np.linspace(0, 1000, 100), fit_curve, label='Fitted Curve', color='#79a55b')
            ax.plot(np.linspace(0, 1000, 100), pred_curve, label='Predicted Curve', linestyle='--', color='black')

        ax.set_title(set_name, fontsize=18)
        #ax.set_xlabel('IPTG (µM)', fontsize=18)
        #ax.set_ylabel('Output (RPU)', fontsize=18)
        ax.tick_params(axis='both', labelsize=18)
        ax.set_xticks([0, 1000])
        max_y_value = myround(max(fit_curve))
        ax.set_yticks([max_y_value])
        #ax.legend()
        #ax.set_ylim([0, 1.25])

    # Remove any empty subplots if the number of sets is odd
    if num_sets % 2 != 0:
        fig.delaxes(axes[-1])

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def HybridModel(x, a, g, beta, Kdb, n, nb):      
    return a * (beta + (1-beta) * 1 / (1 + (g * (x**nb / (Kdb**nb+x**nb)))**n))

x = np.linspace(0, 1000, 100)
Kdb = 165.866479
nb = 1.52072006

fit_QacRQ1_pf5 = result_simultaneous_fit_QacR_pf5.params['a_0'], result_simultaneous_fit_QacR_pf5.params['g_0'], result_simultaneous_fit_QacR_pf5.params['beta_0'], result_simultaneous_fit_QacR_pf5.params['Kdb_0'], result_simultaneous_fit_QacR_pf5.params['n_0'], result_simultaneous_fit_QacR_pf5.params['nb_0']
fit_QacRQ2_pf5 = result_simultaneous_fit_QacR_pf5.params['a_1'], result_simultaneous_fit_QacR_pf5.params['g_1'], result_simultaneous_fit_QacR_pf5.params['beta_1'], result_simultaneous_fit_QacR_pf5.params['Kdb_1'], result_simultaneous_fit_QacR_pf5.params['n_1'], result_simultaneous_fit_QacR_pf5.params['nb_1']
fit_SrpRS1_pf5 = result_simultaneous_fit_SrpR_pf5.params['a_0'], result_simultaneous_fit_SrpR_pf5.params['g_0'], result_simultaneous_fit_SrpR_pf5.params['beta_0'], result_simultaneous_fit_SrpR_pf5.params['Kdb_0'], result_simultaneous_fit_SrpR_pf5.params['n_0'], result_simultaneous_fit_SrpR_pf5.params['nb_0']
fit_SrpRS2_pf5 = result_simultaneous_fit_SrpR_pf5.params['a_1'], result_simultaneous_fit_SrpR_pf5.params['g_1'], result_simultaneous_fit_SrpR_pf5.params['beta_1'], result_simultaneous_fit_SrpR_pf5.params['Kdb_1'], result_simultaneous_fit_SrpR_pf5.params['n_1'], result_simultaneous_fit_SrpR_pf5.params['nb_1']
fit_SrpRS4_pf5 = result_simultaneous_fit_SrpR_pf5.params['a_2'], result_simultaneous_fit_SrpR_pf5.params['g_2'], result_simultaneous_fit_SrpR_pf5.params['beta_2'], result_simultaneous_fit_SrpR_pf5.params['Kdb_2'], result_simultaneous_fit_SrpR_pf5.params['n_2'], result_simultaneous_fit_SrpR_pf5.params['nb_2']

fit_curve_QacRQ1_pf5 = HybridModel(x, fit_QacRQ1_pf5[0].value, fit_QacRQ1_pf5[1].value, fit_QacRQ1_pf5[2].value, fit_QacRQ1_pf5[3].value, fit_QacRQ1_pf5[4].value, fit_QacRQ1_pf5[5].value)
fit_curve_QacRQ2_pf5 = HybridModel(x, fit_QacRQ2_pf5[0].value, fit_QacRQ2_pf5[1].value, fit_QacRQ2_pf5[2].value, fit_QacRQ2_pf5[3].value, fit_QacRQ2_pf5[4].value, fit_QacRQ2_pf5[5].value)
fit_curve_SrpRS1_pf5 = HybridModel(x, fit_SrpRS1_pf5[0].value, fit_SrpRS1_pf5[1].value, fit_SrpRS1_pf5[2].value, fit_SrpRS1_pf5[3].value, fit_SrpRS1_pf5[4].value, fit_SrpRS1_pf5[5].value)
fit_curve_SrpRS2_pf5 = HybridModel(x, fit_SrpRS2_pf5[0].value, fit_SrpRS2_pf5[1].value, fit_SrpRS2_pf5[2].value, fit_SrpRS2_pf5[3].value, fit_SrpRS2_pf5[4].value, fit_SrpRS2_pf5[5].value)
fit_curve_SrpRS4_pf5 = HybridModel(x, fit_SrpRS4_pf5[0].value, fit_SrpRS4_pf5[1].value, fit_SrpRS4_pf5[2].value, fit_SrpRS4_pf5[3].value, fit_SrpRS4_pf5[4].value, fit_SrpRS4_pf5[5].value)


fit_QacRQ1_KT = result_simultaneous_fit_QacR_KT.params['a_0'], result_simultaneous_fit_QacR_KT.params['g_0'], result_simultaneous_fit_QacR_KT.params['beta_0'], result_simultaneous_fit_QacR_KT.params['Kdb_0'], result_simultaneous_fit_QacR_KT.params['n_0'], result_simultaneous_fit_QacR_KT.params['nb_0']
fit_QacRQ2_KT = result_simultaneous_fit_QacR_KT.params['a_1'], result_simultaneous_fit_QacR_KT.params['g_1'], result_simultaneous_fit_QacR_KT.params['beta_1'], result_simultaneous_fit_QacR_KT.params['Kdb_1'], result_simultaneous_fit_QacR_KT.params['n_1'], result_simultaneous_fit_QacR_KT.params['nb_1']
fit_SrpRS1_KT = result_simultaneous_fit_SrpR_KT.params['a_0'], result_simultaneous_fit_SrpR_KT.params['g_0'], result_simultaneous_fit_SrpR_KT.params['beta_0'], result_simultaneous_fit_SrpR_KT.params['Kdb_0'], result_simultaneous_fit_SrpR_KT.params['n_0'], result_simultaneous_fit_SrpR_KT.params['nb_0']
fit_SrpRS2_KT = result_simultaneous_fit_SrpR_KT.params['a_1'], result_simultaneous_fit_SrpR_KT.params['g_1'], result_simultaneous_fit_SrpR_KT.params['beta_1'], result_simultaneous_fit_SrpR_KT.params['Kdb_1'], result_simultaneous_fit_SrpR_KT.params['n_1'], result_simultaneous_fit_SrpR_KT.params['nb_1']
fit_SrpRS4_KT = result_simultaneous_fit_SrpR_KT.params['a_2'], result_simultaneous_fit_SrpR_KT.params['g_2'], result_simultaneous_fit_SrpR_KT.params['beta_2'], result_simultaneous_fit_SrpR_KT.params['Kdb_2'], result_simultaneous_fit_SrpR_KT.params['n_2'], result_simultaneous_fit_SrpR_KT.params['nb_2']

fit_curve_QacRQ1_KT = HybridModel(x, fit_QacRQ1_KT[0].value, fit_QacRQ1_KT[1].value, fit_QacRQ1_KT[2].value, fit_QacRQ1_KT[3].value, fit_QacRQ1_KT[4].value, fit_QacRQ1_KT[5].value)
fit_curve_QacRQ2_KT = HybridModel(x, fit_QacRQ2_KT[0].value, fit_QacRQ2_KT[1].value, fit_QacRQ2_KT[2].value, fit_QacRQ2_KT[3].value, fit_QacRQ2_KT[4].value, fit_QacRQ2_KT[5].value)
fit_curve_SrpRS1_KT = HybridModel(x, fit_SrpRS1_KT[0].value, fit_SrpRS1_KT[1].value, fit_SrpRS1_KT[2].value, fit_SrpRS1_KT[3].value, fit_SrpRS1_KT[4].value, fit_SrpRS1_KT[5].value)
fit_curve_SrpRS2_KT = HybridModel(x, fit_SrpRS2_KT[0].value, fit_SrpRS2_KT[1].value, fit_SrpRS2_KT[2].value, fit_SrpRS2_KT[3].value, fit_SrpRS2_KT[4].value, fit_SrpRS2_KT[5].value)
fit_curve_SrpRS4_KT = HybridModel(x, fit_SrpRS4_KT[0].value, fit_SrpRS4_KT[1].value, fit_SrpRS4_KT[2].value, fit_SrpRS4_KT[3].value, fit_SrpRS4_KT[4].value, fit_SrpRS4_KT[5].value)


def calculate_params_curve(slope, intercept, popt, params, x, Kdb, nb):
    if isinstance(params, dict):
        a = slope[0] * params['a'].value + intercept[0]
        g = popt[0] * np.exp(popt[1] * params['g'].value) + popt[2]
        beta = slope[2] * params['beta'].value + intercept[2]
        n = slope[3] * params['n'].value + intercept[3]
    elif isinstance(params, tuple):
        a = slope[0] * params[0].value + intercept[0]
        g = popt[0] * np.exp(popt[1] * params[1].value) + popt[2]
        beta = slope[2] * params[2].value + intercept[2]
        n = slope[3] * params[4].value + intercept[3]
    else:
        raise ValueError("Invalid params type")

    pred_curve = HybridModel(x, a, g, beta, Kdb, n, nb)
    return pred_curve


pred_curve_LitRL1_pf5 = calculate_params_curve(all_slopes['LitRL1'], all_intercepts['LitRL1'], all_popts['LitRL1'], params_LitRL1_KT, x, Kdb, nb)
pred_curve_HlyIIRH1_pf5 = calculate_params_curve(all_slopes['HlyIIRH1'], all_intercepts['HlyIIRH1'], all_popts['HlyIIRH1'], params_HlyIIRH1_KT, x, Kdb, nb)
pred_curve_BetIE1_pf5 = calculate_params_curve(all_slopes['BetIE1'], all_intercepts['BetIE1'], all_popts['BetIE1'], params_BetIE1_KT, x, Kdb, nb)
pred_curve_lcaRAI1_pf5 = calculate_params_curve(all_slopes['lcaRAI1'], all_intercepts['lcaRAI1'], all_popts['lcaRAI1'], params_lcaRAI1_KT, x, Kdb, nb)
pred_curve_LmrAN1_pf5 = calculate_params_curve(all_slopes['LmrAN1'], all_intercepts['LmrAN1'], all_popts['LmrAN1'], params_LmrAN1_KT, x, Kdb, nb)
pred_curve_QacRQ1_pf5 = calculate_params_curve(all_slopes['QacRQ1'], all_intercepts['QacRQ1'], all_popts['QacRQ1'], fit_QacRQ1_KT, x, Kdb, nb)
pred_curve_QacRQ2_pf5 = calculate_params_curve(all_slopes['QacRQ2'], all_intercepts['QacRQ2'], all_popts['QacRQ2'], fit_QacRQ2_KT, x, Kdb, nb)
pred_curve_SrpRS1_pf5 = calculate_params_curve(all_slopes['SrpRS1'], all_intercepts['SrpRS1'], all_popts['SrpRS1'], fit_SrpRS1_KT, x, Kdb, nb)
pred_curve_SrpRS2_pf5 = calculate_params_curve(all_slopes['SrpRS2'], all_intercepts['SrpRS2'], all_popts['SrpRS2'], fit_SrpRS2_KT, x, Kdb, nb)
pred_curve_SrpRS4_pf5 = calculate_params_curve(all_slopes['SrpRS4'], all_intercepts['SrpRS4'], all_popts['SrpRS4'], fit_SrpRS4_KT, x, Kdb, nb)

plot_fits_prediction(['LitRL1', 'HlyIIRH1', 'BetIE1', 'lcaRAI1', 'LmrAN1', 'QacRQ1', 'QacRQ2', 'SrpRS1', 'SrpRS2', 'SrpRS4'],
                     data = [mean_LitRL1, mean_HlyIIRH1, mean_BetIE1, mean_lcaRAI1, mean_LmrAN1, mean_QacRQ1, mean_QacRQ2, mean_SrpRS1, mean_SrpRS2, mean_SrpRS4],
                     errors = [std_error_LitRL1, std_error_HlyIIRH1, std_error_BetIE1, 
                                 std_error_lcaRAI1, std_error_LmrAN1, std_error_QacRQ1, std_error_QacRQ2, std_error_SrpRS1, std_error_SrpRS2, std_error_SrpRS4],
                     fit_curves=[fit_curve_LitRL1_pf5, fit_curve_HlyIIRH1_pf5, fit_curve_BetIE1_pf5, 
                                 fit_curve_lcaRAI1_pf5, fit_curve_LmrAN1_pf5, fit_curve_QacRQ1_pf5, fit_curve_QacRQ2_pf5, fit_curve_SrpRS1_pf5, fit_curve_SrpRS2_pf5, fit_curve_SrpRS4_pf5],
                     pred_curves=[pred_curve_LitRL1_pf5, pred_curve_HlyIIRH1_pf5, pred_curve_BetIE1_pf5, 
                                 pred_curve_lcaRAI1_pf5, pred_curve_LmrAN1_pf5, pred_curve_QacRQ1_pf5, pred_curve_QacRQ2_pf5, pred_curve_SrpRS1_pf5, pred_curve_SrpRS2_pf5, pred_curve_SrpRS4_pf5])



#%% Prediction metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Assuming you have actual data in actual_data variable
actual_data = [fit_curve_LitRL1_pf5, fit_curve_HlyIIRH1_pf5, fit_curve_BetIE1_pf5, 
               fit_curve_lcaRAI1_pf5, fit_curve_LmrAN1_pf5, fit_curve_QacRQ1_pf5,
               fit_curve_QacRQ2_pf5, fit_curve_SrpRS1_pf5, fit_curve_SrpRS2_pf5, fit_curve_SrpRS4_pf5] 

# Calculate predictions for the corresponding sets
predictions = [pred_curve_LitRL1_pf5, pred_curve_HlyIIRH1_pf5, pred_curve_BetIE1_pf5, 
               pred_curve_lcaRAI1_pf5, pred_curve_LmrAN1_pf5, pred_curve_QacRQ1_pf5, 
               pred_curve_QacRQ2_pf5, pred_curve_SrpRS1_pf5, pred_curve_SrpRS2_pf5, pred_curve_SrpRS4_pf5] 

names = ['LitRL1', 'HlyIIRH1', 'BetIE1', 'lcaRAI1', 'LmrAN1', 'QacRQ1', 'QacRQ2', 'SrpRS1', 'SrpRS2', 'SrpRS4']

data = []
r2_all = []

for actual_datum, prediction, name in zip(actual_data, predictions, names):
    mse = mean_squared_error(actual_datum, prediction)
    rmse = mean_squared_error(actual_datum, prediction, squared=False)
    mae = mean_absolute_error(actual_datum, prediction)
    r2 = r2_score(actual_datum, prediction)

    row = {
        'Name': name,
        'R² Score': r2,
        'Mean Squared Error (MSE)': mse,
        'Root Mean Squared Error (RMSE)': rmse,
        'Mean Absolute Error (MAE)': mae
    }

    data.append(row)
    r2_all.append(r2)

# Create a DataFrame
df = pd.DataFrame(data)

# Print the DataFrame
print(df)

# Export to Excel
df.to_excel('metrics_results.xlsx', index=False)


mean_r2 = np.mean(r2_all)
std_r2 = np.std(r2_all)

print('mean_r2 =', mean_r2)
print('std_r2 =', std_r2)



#%% Random Prediction Validation
def calculate_params_curve_random(slope, intercept, popt, params, x, Kdb, nb):
    a = np.random.rand() * 2
    g = np.random.rand() * 200
    beta = np.random.rand() * 1
    n = np.random.rand() * 2.5

    pred_curve = HybridModel(x, a, g, beta, Kdb, n, nb)
    return pred_curve


evaluation_r2 = []
num_repetitions = 1000

for _ in range(num_repetitions):
    pred_curve_LitRL1_pf5_rdn = calculate_params_curve_random(all_slopes['LitRL1'], all_intercepts['LitRL1'], all_popts['LitRL1'], params_LitRL1_KT, x, Kdb, nb)
    pred_curve_HlyIIRH1_pf5_rdn = calculate_params_curve_random(all_slopes['HlyIIRH1'], all_intercepts['HlyIIRH1'], all_popts['HlyIIRH1'], params_HlyIIRH1_KT, x, Kdb, nb)
    pred_curve_BetIE1_pf5_rdn = calculate_params_curve_random(all_slopes['BetIE1'], all_intercepts['BetIE1'], all_popts['BetIE1'], params_BetIE1_KT, x, Kdb, nb)
    pred_curve_lcaRAI1_pf5_rdn = calculate_params_curve_random(all_slopes['lcaRAI1'], all_intercepts['lcaRAI1'], all_popts['lcaRAI1'], params_lcaRAI1_KT, x, Kdb, nb)
    pred_curve_LmrAN1_pf5_rdn = calculate_params_curve_random(all_slopes['LmrAN1'], all_intercepts['LmrAN1'], all_popts['LmrAN1'], params_LmrAN1_KT, x, Kdb, nb)
    pred_curve_QacRQ1_pf5_rdn = calculate_params_curve_random(all_slopes['QacRQ1'], all_intercepts['QacRQ1'], all_popts['QacRQ1'], fit_QacRQ1_KT, x, Kdb, nb)
    pred_curve_QacRQ2_pf5_rdn = calculate_params_curve_random(all_slopes['QacRQ2'], all_intercepts['QacRQ2'], all_popts['QacRQ2'], fit_QacRQ2_KT, x, Kdb, nb)
    pred_curve_SrpRS1_pf5_rdn = calculate_params_curve_random(all_slopes['SrpRS1'], all_intercepts['SrpRS1'], all_popts['SrpRS1'], fit_SrpRS1_KT, x, Kdb, nb)
    pred_curve_SrpRS2_pf5_rdn = calculate_params_curve_random(all_slopes['SrpRS2'], all_intercepts['SrpRS2'], all_popts['SrpRS2'], fit_SrpRS2_KT, x, Kdb, nb)
    pred_curve_SrpRS4_pf5_rdn = calculate_params_curve_random(all_slopes['SrpRS4'], all_intercepts['SrpRS4'], all_popts['SrpRS4'], fit_SrpRS4_KT, x, Kdb, nb)

    # Actual data in actual_data variable
    actual_data_all = [fit_curve_LitRL1_pf5, fit_curve_HlyIIRH1_pf5, fit_curve_BetIE1_pf5, 
                fit_curve_lcaRAI1_pf5, fit_curve_LmrAN1_pf5, fit_curve_QacRQ1_pf5,
                fit_curve_QacRQ2_pf5, fit_curve_SrpRS1_pf5, fit_curve_SrpRS2_pf5, fit_curve_SrpRS4_pf5] 

    # Calculate predictions for the corresponding sets
    predictions_rdn = [pred_curve_LitRL1_pf5_rdn, pred_curve_HlyIIRH1_pf5_rdn, pred_curve_BetIE1_pf5_rdn, 
                pred_curve_lcaRAI1_pf5_rdn, pred_curve_LmrAN1_pf5_rdn, pred_curve_QacRQ1_pf5_rdn, 
                pred_curve_QacRQ2_pf5_rdn, pred_curve_SrpRS1_pf5_rdn, pred_curve_SrpRS2_pf5_rdn, pred_curve_SrpRS4_pf5_rdn] 

    names_all = ['LitRL1', 'HlyIIRH1', 'BetIE1', 'lcaRAI1', 'LmrAN1', 'QacRQ1', 'QacRQ2', 'SrpRS1', 'SrpRS2', 'SrpRS4']

    data = []

    for actual_datum_all, prediction_rdn, name_all in zip(actual_data_all, predictions_rdn, names_all):
        r2 = r2_score(actual_datum_all, prediction_rdn)

    evaluation_r2.append(r2)

mean_r2_random = np.mean(evaluation_r2)
std_r2_random = np.std(evaluation_r2)

print('mean_r2_random =', mean_r2_random)
print('std_r2_random =', std_r2_random)



# Plot boxplots
plt.figure(figsize=(6, 5))
plt.boxplot([evaluation_r2, r2_all], labels=['Random Predictions', 'Model'])
plt.ylabel('R² Values')
#plt.title('Comparison of R² Values')
plt.grid(True)
plt.show()
#%% Probability of achieving model performance by chance
# Calculate the proportion of random R² values equal to or greater than model's R² value
probability_by_chance = np.mean(np.array(evaluation_r2) >= mean_r2)

p = (np.array(evaluation_r2) >= mean_r2)
q = np.array(evaluation_r2)

# Convert to percentage
probability_by_chance_percentage = probability_by_chance * 100

print('Probability of achieving model performance by chance: {:.2f}%'.format(probability_by_chance_percentage))




#%% Prediction metrics - Excluding bad gates
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Assuming you have actual data in actual_data variable
actual_data = [fit_curve_LitRL1_pf5,
               fit_curve_lcaRAI1_pf5, fit_curve_LmrAN1_pf5, fit_curve_QacRQ1_pf5,
               fit_curve_QacRQ2_pf5, fit_curve_SrpRS1_pf5, fit_curve_SrpRS2_pf5, fit_curve_SrpRS4_pf5] 

# Calculate predictions for the corresponding sets
predictions = [pred_curve_LitRL1_pf5,
               pred_curve_lcaRAI1_pf5, pred_curve_LmrAN1_pf5, pred_curve_QacRQ1_pf5, 
               pred_curve_QacRQ2_pf5, pred_curve_SrpRS1_pf5, pred_curve_SrpRS2_pf5, pred_curve_SrpRS4_pf5] 

names = ['LitRL1', 'lcaRAI1', 'LmrAN1', 'QacRQ1', 'QacRQ2', 'SrpRS1', 'SrpRS2', 'SrpRS4']

data = []
r2_all = []

for actual_datum, prediction, name in zip(actual_data, predictions, names):
    mse = mean_squared_error(actual_datum, prediction)
    rmse = mean_squared_error(actual_datum, prediction, squared=False)
    mae = mean_absolute_error(actual_datum, prediction)
    r2 = r2_score(actual_datum, prediction)

    row = {
        'Name': name,
        'R² Score': r2,
        'Mean Squared Error (MSE)': mse,
        'Root Mean Squared Error (RMSE)': rmse,
        'Mean Absolute Error (MAE)': mae
    }

    data.append(row)
    r2_all.append(r2)

# Create a DataFrame
df = pd.DataFrame(data)

# Print the DataFrame
print(df)

# Export to Excel
df.to_excel('metrics_results.xlsx', index=False)


mean_r2 = np.mean(r2_all)
std_r2 = np.std(r2_all)

print('mean_r2 =', mean_r2)
print('std_r2 =', std_r2)

#%% Random Prediction Validation - Excluding bad gates
def calculate_params_curve_random(slope, intercept, popt, params, x, Kdb, nb):
    a = np.random.rand() * 2
    g = np.random.rand() * 200
    beta = np.random.rand() * 1
    n = np.random.rand() * 2.5

    pred_curve = HybridModel(x, a, g, beta, Kdb, n, nb)
    return pred_curve


evaluation_r2 = []
num_repetitions = 1000

for _ in range(num_repetitions):
    pred_curve_LitRL1_pf5_rdn = calculate_params_curve_random(all_slopes['LitRL1'], all_intercepts['LitRL1'], all_popts['LitRL1'], params_LitRL1_KT, x, Kdb, nb)
    pred_curve_lcaRAI1_pf5_rdn = calculate_params_curve_random(all_slopes['lcaRAI1'], all_intercepts['lcaRAI1'], all_popts['lcaRAI1'], params_lcaRAI1_KT, x, Kdb, nb)
    pred_curve_LmrAN1_pf5_rdn = calculate_params_curve_random(all_slopes['LmrAN1'], all_intercepts['LmrAN1'], all_popts['LmrAN1'], params_LmrAN1_KT, x, Kdb, nb)
    pred_curve_QacRQ1_pf5_rdn = calculate_params_curve_random(all_slopes['QacRQ1'], all_intercepts['QacRQ1'], all_popts['QacRQ1'], fit_QacRQ1_KT, x, Kdb, nb)
    pred_curve_QacRQ2_pf5_rdn = calculate_params_curve_random(all_slopes['QacRQ2'], all_intercepts['QacRQ2'], all_popts['QacRQ2'], fit_QacRQ2_KT, x, Kdb, nb)
    pred_curve_SrpRS1_pf5_rdn = calculate_params_curve_random(all_slopes['SrpRS1'], all_intercepts['SrpRS1'], all_popts['SrpRS1'], fit_SrpRS1_KT, x, Kdb, nb)
    pred_curve_SrpRS2_pf5_rdn = calculate_params_curve_random(all_slopes['SrpRS2'], all_intercepts['SrpRS2'], all_popts['SrpRS2'], fit_SrpRS2_KT, x, Kdb, nb)
    pred_curve_SrpRS4_pf5_rdn = calculate_params_curve_random(all_slopes['SrpRS4'], all_intercepts['SrpRS4'], all_popts['SrpRS4'], fit_SrpRS4_KT, x, Kdb, nb)

    # Actual data in actual_data variable
    actual_data = [fit_curve_LitRL1_pf5, 
                fit_curve_lcaRAI1_pf5, fit_curve_LmrAN1_pf5, fit_curve_QacRQ1_pf5,
                fit_curve_QacRQ2_pf5, fit_curve_SrpRS1_pf5, fit_curve_SrpRS2_pf5, fit_curve_SrpRS4_pf5] 

    # Calculate predictions for the corresponding sets
    predictions_rdn = [pred_curve_LitRL1_pf5_rdn, 
                pred_curve_lcaRAI1_pf5_rdn, pred_curve_LmrAN1_pf5_rdn, pred_curve_QacRQ1_pf5_rdn, 
                pred_curve_QacRQ2_pf5_rdn, pred_curve_SrpRS1_pf5_rdn, pred_curve_SrpRS2_pf5_rdn, pred_curve_SrpRS4_pf5_rdn] 

    names = ['LitRL1', 'lcaRAI1', 'LmrAN1', 'QacRQ1', 'QacRQ2', 'SrpRS1', 'SrpRS2', 'SrpRS4']

    data = []

    for actual_datum, prediction, name in zip(actual_data, predictions_rdn, names):
        r2 = r2_score(actual_datum, prediction)

    evaluation_r2.append(r2)

mean_r2_random = np.mean(evaluation_r2)
std_r2_random = np.std(evaluation_r2)

print('mean_r2_random =', mean_r2_random)
print('std_r2_random =', std_r2_random)



# Plot boxplots
plt.figure(figsize=(6, 5))
plt.boxplot([evaluation_r2, r2_all], labels=['Random Predictions', 'Model'])
plt.ylabel('R² Values')
#plt.title('Comparison of R² Values')
plt.grid(True)
plt.show()
#%% - Excluding bad gates
# Calculate the proportion of random R² values equal to or greater than model's R² value
probability_by_chance = np.mean(np.array(evaluation_r2) >= mean_r2)

p = (np.array(evaluation_r2) >= mean_r2)
q = np.array(evaluation_r2)

# Convert to percentage
probability_by_chance_percentage = probability_by_chance * 100

print('Probability of achieving model performance by chance: {:.2f}%'.format(probability_by_chance_percentage))


#%% Plot fits only bars

######################## pf5 #########################
# Regular gates
plot_fits_onlyBars([mean_1818, mean_LitRL1, mean_HlyIIRH1, mean_BetIE1, mean_lcaRAI1, mean_LmrAN1],
                     ['1818', 'LitRL1', 'HlyIIRH1', 'BetIE1', 'lcaRAI1', 'LmrAN1'],
                     errors=[std_error_1818, std_error_LitRL1, std_error_HlyIIRH1, std_error_BetIE1, 
                             std_error_lcaRAI1, std_error_LmrAN1],
                     fit_curves=[fit_curve_1818_pf5, fit_curve_LitRL1_pf5, fit_curve_HlyIIRH1_pf5, fit_curve_BetIE1_pf5, 
                                 fit_curve_lcaRAI1_pf5, fit_curve_LmrAN1_pf5],
                     results=[result_Control_pf5.params, result_LitRL1_pf5.params, result_HlyIIRH1_pf5.params, result_BetIE1_pf5.params,
                               result_lcaRAI1_pf5.params, result_LmrAN1_pf5.params], alpha=0.6)
# Simultaneous gates
plot_fits_onlyBars([mean_1818, mean_QacRQ1, mean_QacRQ2, mean_SrpRS1, mean_SrpRS2, mean_SrpRS4],
                     ['1818', 'QacRQ1', 'QacRQ2', 'SrpRS1', 'SrpRS2', 'SrpRS4'],
                     errors=[std_error_1818, std_error_QacRQ1, std_error_QacRQ2, std_error_SrpRS1, std_error_SrpRS2, std_error_SrpRS4],
                     fit_curves=[fit_curve_1818_pf5, fit_curve_QacRQ1_pf5, fit_curve_QacRQ2_pf5, fit_curve_SrpRS1_pf5, 
                                 fit_curve_SrpRS2_pf5, fit_curve_SrpRS4_pf5],
                     results=[result_Control_pf5.params, result_QacRQ1_params_pf5, result_QacRQ2_params_pf5, result_SrpRS1_params_pf5
                              , result_SrpRS2_params_pf5, result_SrpRS4_params_pf5], alpha=0.6)

######################## KT #########################
# Regular gates
plot_fits_onlyBars([df_1818, df_LitRL1, df_HlyIIRH1, df_BetIE1, df_lcaRAI1, df_LmrAN1],
                     ['1818', 'LitRL1', 'HlyIIRH1', 'BetIE1', 'lcaRAI1', 'LmrAN1'],
                     errors=[None, None, None, None, None, None, None],
                     fit_curves=[fit_curve_1818_KT, fit_curve_LitRL1_KT, fit_curve_HlyIIRH1_KT, fit_curve_BetIE1_KT, 
                                 fit_curve_lcaRAI1_KT, fit_curve_LmrAN1_KT],
                     results=[result_Control_KT.params, result_LitRL1_KT.params, result_HlyIIRH1_KT.params, result_BetIE1_KT.params,
                               result_lcaRAI1_KT.params, result_LmrAN1_KT.params], alpha=0.6)
# Simultaneous gates
plot_fits_onlyBars([df_1818, df_QacRQ1, df_QacRQ2, df_SrpRS1, df_SrpRS2, df_SrpRS4],
                     ['1818', 'QacRQ1', 'QacRQ2', 'SrpRS1', 'SrpRS2', 'SrpRS4'],
                     errors=[None, None, None, None, None, None],
                     fit_curves=[fit_curve_1818_KT, fit_curve_QacRQ1_KT, fit_curve_QacRQ2_KT, fit_curve_SrpRS1_KT, 
                                 fit_curve_SrpRS2_KT, fit_curve_SrpRS4_KT],
                     results=[result_Control_KT.params, result_QacRQ1_params_KT, result_QacRQ2_params_KT, result_SrpRS1_params_KT
                              , result_SrpRS2_params_KT, result_SrpRS4_params_KT], alpha=0.6)


#%% Influence of each parameter 
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 1000, 500)
a = 1 
g = 10
beta = 0
Kdc = 500
nc = 1
n = 1

# Set a common color for all plots
lineColor_a = [0.58, 0.404, 0.741]
lineColor_g = [0.737, 0.741, 0.133]
lineColor_beta = [0.498, 0.498, 0.498]
lineColor_n = [0.09, 0.745, 0.812]
alphas = np.linspace(1, 0.2, 5)

# Parameters to systematically vary
a_values = [2.5, 1.5, 1, 0.5, 0.1]
g_values = [150, 50, 15, 5, 2]
beta_values = [0.8, 0.6, 0.4, 0.2, 0]
n_values = [2.5, 1.5, 1, 0.5, 0.1]



plt.figure()
for j, a_value in enumerate(a_values):
    # Initialize an array to store the results
    expression_result = np.zeros_like(x)

    # Evaluate the expression for each value of x
    for i in range(len(x)):
        expression_result[i] = a_value * (beta + (1 - beta) * 1 / (1 + (g * x[i]**nc / (Kdc**nc + x[i]**nc))**n))

    # Plot the expression for the current value of g with descending alpha
    plt.plot(x, expression_result, label=f'g = {g}', linewidth=3, color=lineColor_a, alpha = alphas[j])

plt.xlim([0, 1000])
plt.ylim([0, 2.6])
plt.xticks([])
plt.yticks([])
plt.xlabel('IPTG (µM)', fontsize=20)
plt.ylabel('Output (RPU)', fontsize=20)
plt.show()


plt.figure()
for j, g_value in enumerate(g_values):
    # Initialize an array to store the results
    expression_result = np.zeros_like(x)

    # Evaluate the expression for each value of x
    for i in range(len(x)):
        expression_result[i] = a * (beta + (1 - beta) * 1 / (1 + (g_value * x[i]**nc / (Kdc**nc + x[i]**nc))**n))

    # Plot the expression for the current value of g with varying alpha
    plt.plot(x, expression_result, label=f'g = {g_value}', linewidth=3, color=lineColor_g, alpha = alphas[j])

plt.xlim([0, 1000])
plt.ylim([0, 1.05])
plt.xticks([])
plt.yticks([])
plt.xlabel('IPTG (µM)', fontsize=20)
plt.ylabel('Output (RPU)', fontsize=20)
plt.show()


plt.figure()
for j, beta_value in enumerate(beta_values):
    # Initialize an array to store the results
    expression_result = np.zeros_like(x)

    # Evaluate the expression for each value of x
    for i in range(len(x)):
        expression_result[i] = a * (beta_value + (1 - beta_value) * 1 / (1 + (g * x[i]**nc / (Kdc**nc + x[i]**nc))**n))

    # Plot the expression for the current value of beta with descending alpha
    plt.plot(x, expression_result, label=f'beta = {beta}', linewidth=3, color=lineColor_beta, alpha = alphas[j])

plt.xlim([0, 1000])
plt.ylim([0, 1.05])
plt.xticks([])
plt.yticks([])
plt.xlabel('IPTG (µM)', fontsize=20)
plt.ylabel('Output (RPU)', fontsize=20)
plt.show()


plt.figure()
for j, n_value in enumerate(n_values):
    # Initialize an array to store the results
    expression_result = np.zeros_like(x)

    # Evaluate the expression for each value of x
    for i in range(len(x)):
        expression_result[i] = a * (beta + (1 - beta) * 1 / (1 + (g * x[i]**nc / (Kdc**nc + x[i]**nc))**n_value))

    # Plot the expression for the current value of n with descending alpha
    plt.plot(x, expression_result, label=f'n = {n}', linewidth=3, color=lineColor_n, alpha = alphas[j])

plt.xlim([0, 1000])
plt.ylim([0, 1.05])
plt.xticks([])
plt.yticks([])
plt.xlabel('IPTG (µM)', fontsize=20)
plt.ylabel('Output (RPU)', fontsize=20)
plt.show()




















#%% Plot fits pf5 and KT

def plot_fits_combined(hosts, mean_rows_list, set_names, errors_list=None, fit_curves_list=None, figure_size=(1.8, 2.2)):

    num_sets = len(set_names)
    num_rows, num_cols = divmod(num_sets, 1)
    fig, axes = plt.subplots(num_rows + (1 if num_cols > 0 else 0), 1, figsize=(figure_size[0] * 2, figure_size[1] * num_rows))

    
    # Adjust the spacing between subplots
    plt.subplots_adjust(wspace=0.6)
    
    for host, mean_rows, errors, fit_curves in zip(hosts, mean_rows_list, errors_list, fit_curves_list):
        if host == 'pf5':
            IPTG = pd.Series([10, 20, 30, 40, 50, 70, 100, 200, 500, 1000])
            color = '#79a55b'
        elif host == 'KT':
            IPTG = pd.Series([0, 5, 10, 20, 30, 40, 50, 70, 100, 200, 500, 1000])
            color = '#d9798f'
        else:
            raise ValueError(f"Unsupported host: {host}")

        for ax, mean_row, set_name, error, fit_curve in zip(axes.flatten(), mean_rows, set_names, errors, fit_curves):
            if host == 'pf5' and set_name == 'QacRQ1':
                IPTG = pd.Series([20, 30, 40, 50, 70, 100, 200, 500, 1000])

            if error is not None:
                error = np.asarray(error).flatten()
                ax.errorbar(IPTG, mean_row, yerr=error, label=f'Mean {set_name} - {host}', marker='o', linestyle='None', color=color, markersize=5)
            else:
                ax.plot(IPTG, mean_row, label=f'Mean {set_name} - {host}', marker='o', linestyle='None', color=color, markersize=5)

            if fit_curve is not None:
                ax.plot(np.linspace(0, 1000, 100), fit_curve, label=f'Fitted Curve - {host}', color=color, linewidth=1.5)

            ax.set_title(set_name, fontsize=18)
            ax.set_xticks([])
            ax.tick_params(axis='both', labelsize=18)
            if set_name == '1818':
                ax.set_yticks([0.7])
            else:
                max_value = myround(max(fit_curve)+ 0.2 * max(fit_curve))
                ax.set_yticks([max_value])

            ax.set_ylim(bottom=0)

    #plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

plot_fits_combined(['pf5', 'KT'],
                   [[mean_1818, mean_LitRL1, mean_HlyIIRH1, mean_BetIE1, mean_lcaRAI1, mean_LmrAN1, mean_QacRQ1, mean_QacRQ2, mean_SrpRS1, mean_SrpRS2, mean_SrpRS4],
                    [df_1818, df_LitRL1, df_HlyIIRH1, df_BetIE1, df_lcaRAI1, df_LmrAN1, df_QacRQ1, df_QacRQ2, df_SrpRS1, df_SrpRS2, df_SrpRS4]],
                   ['1818', 'LitRL1', 'HlyIIRH1', 'BetIE1', 'lcaRAI1', 'LmrAN1', 'QacRQ1', 'QacRQ2', 'SrpRS1', 'SrpRS2', 'SrpRS4'],
                   errors_list=[[std_error_1818, std_error_LitRL1, std_error_HlyIIRH1, std_error_BetIE1, 
                                 std_error_lcaRAI1, std_error_LmrAN1, std_error_QacRQ1, std_error_QacRQ2, std_error_SrpRS1, std_error_SrpRS2, std_error_SrpRS4],
                                [None, None, None, None, None, None, None, None, None, None, None]],
                   fit_curves_list=[[fit_curve_1818_pf5, fit_curve_LitRL1_pf5, fit_curve_HlyIIRH1_pf5, fit_curve_BetIE1_pf5, 
                                     fit_curve_lcaRAI1_pf5, fit_curve_LmrAN1_pf5, fit_curve_QacRQ1_pf5, fit_curve_QacRQ2_pf5, fit_curve_SrpRS1_pf5, fit_curve_SrpRS2_pf5, fit_curve_SrpRS4_pf5],
                                    [fit_curve_1818_KT, fit_curve_LitRL1_KT, fit_curve_HlyIIRH1_KT, fit_curve_BetIE1_KT, 
                                     fit_curve_lcaRAI1_KT, fit_curve_LmrAN1_KT, fit_curve_QacRQ1_KT, fit_curve_QacRQ2_KT, fit_curve_SrpRS1_KT, fit_curve_SrpRS2_KT, fit_curve_SrpRS4_KT]])



#%% calculate_ratio_and_error
import numpy as np

def calculate_ratio_and_error(value1, error1, value2, error2, operation='division'):
    # Calculate the ratio
    if operation == 'division':
        ratio = value1 / value2
        # Calculate the standard error for the ratio
        ratio_error = np.sqrt((1 / np.abs(value2) * error1)**2 + 
                              (np.abs(value1) / np.abs(value2)**2 * error2)**2)
    else:
        # Handle other operations if needed
        print("Unsupported operation")
        return None
    
    return ratio, ratio_error

# Example usage:

# Values and standard errors for Q1_Q2_pf5
value_Q1_pf5 = result_QacRQ1_params_pf5['g'].value
error_Q1_pf5 = result_QacRQ1_params_pf5['g'].stderr
value_Q2_pf5 = result_QacRQ2_params_pf5['g'].value
error_Q2_pf5 = result_QacRQ2_params_pf5['g'].stderr

# Calculate Q1_Q2_pf5 ratio and standard error
Q1_Q2_pf5, SE_Q1_Q2_pf5 = calculate_ratio_and_error(value_Q1_pf5, error_Q1_pf5, value_Q2_pf5, error_Q2_pf5)

# Values and standard errors for Q1_Q2_KT
value_Q1_KT = result_QacRQ1_params_KT['g'].value
error_Q1_KT = result_QacRQ1_params_KT['g'].stderr
value_Q2_KT = result_QacRQ2_params_KT['g'].value
error_Q2_KT = result_QacRQ2_params_KT['g'].stderr

# Calculate Q1_Q2_KT ratio and standard error
Q1_Q2_KT, SE_Q1_Q2_KT = calculate_ratio_and_error(value_Q1_KT, error_Q1_KT, value_Q2_KT, error_Q2_KT)

# Print the results
print("Q1/Q2_pf5:", Q1_Q2_pf5)
print("Standard Error for Q1/Q2_pf5:", SE_Q1_Q2_pf5)

print("Q1_Q2_KT:", Q1_Q2_KT)
print("Standard Error for Q1/Q2_KT:", SE_Q1_Q2_KT)



# Values and standard errors for S1_S2_pf5
value_S1_pf5 = result_SrpRS1_params_pf5['g'].value
error_S1_pf5 = result_SrpRS1_params_pf5['g'].stderr
value_S2_pf5 = result_SrpRS2_params_pf5['g'].value
error_S2_pf5 = result_SrpRS2_params_pf5['g'].stderr

# Calculate S1_S2_pf5 ratio and standard error
S1_S2_pf5, SE_S1_S2_pf5 = calculate_ratio_and_error(value_S1_pf5, error_S1_pf5, value_S2_pf5, error_S2_pf5)

# Values and standard errors for S1_S2_KT
value_S1_KT = result_SrpRS1_params_KT['g'].value
error_S1_KT = result_SrpRS1_params_KT['g'].stderr
value_S2_KT = result_SrpRS2_params_KT['g'].value
error_S2_KT = result_SrpRS2_params_KT['g'].stderr

# Calculate S1_S2_KT ratio and standard error
S1_S2_KT, SE_S1_S2_KT = calculate_ratio_and_error(value_S1_KT, error_S1_KT, value_S2_KT, error_S2_KT)

# Print the results
print("S1/S2_pf5:", S1_S2_pf5)
print("Standard Error for S1/S2_pf5:", SE_S1_S2_pf5)

print("S1_S2_KT:", S1_S2_KT)
print("Standard Error for S1/S2_KT:", SE_S1_S2_KT)

# Values and standard errors for S2_S4_pf5
value_S2_pf5 = result_SrpRS2_params_pf5['g'].value
error_S2_pf5 = result_SrpRS2_params_pf5['g'].stderr
value_S4_pf5 = result_SrpRS4_params_pf5['g'].value  # Change column if needed
error_S4_pf5 = result_SrpRS4_params_pf5['g'].stderr  # Change column if needed

# Calculate S2_S4_pf5 ratio and standard error
S2_S4_pf5, SE_S2_S4_pf5 = calculate_ratio_and_error(value_S2_pf5, error_S2_pf5, value_S4_pf5, error_S4_pf5)

# Values and standard errors for S2_S4_KT
value_S2_KT = result_SrpRS2_params_KT['g'].value
error_S2_KT = result_SrpRS2_params_KT['g'].stderr
value_S4_KT = result_SrpRS4_params_KT['g'].value  # Change column if needed
error_S4_KT = result_SrpRS4_params_KT['g'].stderr  # Change column if needed

# Calculate S2_S4_KT ratio and standard error
S2_S4_KT, SE_S2_S4_KT = calculate_ratio_and_error(value_S2_KT, error_S2_KT, value_S4_KT, error_S4_KT)

# Print the results
print("S2/S4_pf5:", S2_S4_pf5)
print("Standard Error for S2/S4_pf5:", SE_S2_S4_pf5)

print("S2/S4_KT:", S2_S4_KT)
print("Standard Error for S2/S4_KT:", SE_S2_S4_KT)




# Calculate Q1_Q1 ratio and standard error
Q1_Q1, SE_Q1_Q1 = calculate_ratio_and_error(value_Q1_pf5, error_Q1_pf5, value_Q1_KT, error_Q1_KT)

# Print the results
print("Q1(pf5)/Q1(KT):", Q1_Q1)
print("Standard Error for Q1(pf5)/Q1(KT):", SE_Q1_Q1)

print(value_Q1_pf5, value_Q1_KT)

# Calculate Q2_Q2 ratio and standard error
Q2_Q2, SE_Q2_Q2 = calculate_ratio_and_error(value_Q2_pf5, error_Q2_pf5, value_Q2_KT, error_Q2_KT)

# Print the results
print("Q2(pf5)/Q2(KT):", Q2_Q2)
print("Standard Error for Q2(pf5)/Q2(KT):", SE_Q2_Q2)



# Calculate S1_S1 ratio and standard error
S1_S1, SE_S1_S1 = calculate_ratio_and_error(value_S1_pf5, error_S1_pf5, value_S1_KT, error_S1_KT)

# Print the results
print("S1(pf5)/S1(KT):", S1_S1)
print("Standard Error for S1(pf5)/S1(KT):", SE_S1_S1)

# Calculate S2_S2 ratio and standard error
S2_S2, SE_S2_S2 = calculate_ratio_and_error(value_S2_pf5, error_S2_pf5, value_S2_KT, error_S2_KT)

# Print the results
print("S2(pf5)/S2(KT):", S2_S2)
print("Standard Error for S2(pf5)/S2(KT):", SE_S2_S2)

# Calculate S4_S4 ratio and standard error
S4_S4, SE_S4_S4 = calculate_ratio_and_error(value_S4_pf5, error_S4_pf5, value_S4_KT, error_S4_KT)

# Print the results
print("S4(pf5)/S4(KT):", S4_S4)
print("Standard Error for S4(pf5)/S4(KT):", SE_S4_S4)







