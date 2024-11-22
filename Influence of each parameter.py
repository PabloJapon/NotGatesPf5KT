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




