
import numpy as np
from scipy.stats import chi2

from simulate_data import simulate_conditional_independence, simulate_conditional_dependence, frequency_table

def calculate_expected(observed):
    expected = np.zeros_like(observed)
    for z in range(observed.shape[2]):
        # Marginal totals for X and Y given Z
        sum_X_given_Z = np.sum(observed[:, :, z], axis=1, keepdims=True)
        sum_Y_given_Z = np.sum(observed[:, :, z], axis=0, keepdims=True)
        total_given_Z = np.sum(observed[:, :, z])
        # Expected frequency calculation for each cell under the assumption of conditional independence
        expected[:, :, z] = sum_X_given_Z * sum_Y_given_Z / total_given_Z
    return expected

def chi_squared_test(observed):
    expected = calculate_expected(observed)
    
    # Calculate Chi-squared statistic
    chi_squared_stat = np.sum(((observed - expected) ** 2) / expected)
    
    # Calculate degrees of freedom
    dof = (observed.shape[0] - 1) * (observed.shape[1] - 1) * observed.shape[2]

    p_value = 1 - chi2.cdf(chi_squared_stat, dof)
    
    return chi_squared_stat, p_value, dof

def p_value_conclusion(p_val, alpha=0.05):
    if p_val > alpha:
        print("There is insufficient evidence to reject the null hypothesis of conditional independence between the variables. ")
    else:
        print("There is sufficient evidence to reject the null hypothesis of conditional independence between the variables.(Suggesting Variables are conditonally dependent given C)")

if __name__ == "__main__":
    size = 10000
    data_independence = frequency_table(*simulate_conditional_independence(size))
    data_dependence = frequency_table(*simulate_conditional_dependence(size))


    print("Conditional Independence Data:")
    print(data_independence)


    chi_squared_stat, p_value, dof = chi_squared_test(data_independence)


    print("Independent Data")
    print(f"Chi-squared statistic: {chi_squared_stat}")
    print(f"Degrees of freedom: {dof}")
    print(f"P-value: {p_value:.05}")
    p_value_conclusion(p_value)

    print("\nConditional Dependence Data:")
    print(data_dependence)
    chi_squared_stat, p_value, dof = chi_squared_test(data_dependence)
    print("Dependent Data")
    print(f"Chi-squared statistic: {chi_squared_stat}")
    print(f"Degrees of freedom: {dof}")
    print(f"P-value: {p_value:.05}")
    p_value_conclusion(p_value)