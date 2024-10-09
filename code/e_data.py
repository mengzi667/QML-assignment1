import numpy as np


def get_data():
    """
    Retrieves the data required for the experiment.

    Returns:
        tuple: A tuple containing the following elements:
            - demand (ndarray): Monthly demand for each product.
            - holding_costs (ndarray): Inventory holding costs for each product (Euro/kg).
            - supplier_costs (ndarray): Supplier cost per kilogram of material (Euro).
            - capacity (ndarray): Monthly production capacity limits.
            - supply_limit (ndarray): Maximum supply quantity per supplier each month.
            - Cr (ndarray): Chromium percentages in materials provided by suppliers.
            - Ni (ndarray): Nickel percentages in materials provided by suppliers.
            - Cr_required (ndarray): Required Chromium content for each product.
            - Ni_required (ndarray): Required Nickel content for each product.
            - CopperLimit (ndarray): Monthly copper limit.
            - Cu (ndarray): Copper percentages in materials provided by suppliers.
    """
    n_months, n_products, n_suppliers = 12, 3, 5

    demand = np.array([
        [25, 25, 0, 0, 0, 50, 12, 0, 10, 10, 45, 99],  # 18/10
        [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10],  # 18/8
        [5, 20, 80, 25, 50, 125, 150, 80, 40, 35, 3, 100]  # 18/0
    ])

    holding_costs = np.array([20, 10, 5])
    supplier_costs = np.array([5, 10, 9, 7, 8.5])
    capacity = np.full(n_months, 100)
    supply_limit = np.array([90, 30, 50, 70, 20])
    Cr = np.array([18, 25, 15, 14, 0])
    Ni = np.array([0, 15, 10, 16, 10])
    Cu = np.array([0, 4, 2, 5, 3])
    Cr_required = np.array([18, 18, 18])
    Ni_required = np.array([10, 8, 0])
    CopperLimit = np.full(n_months, 1.39)

    return demand, holding_costs, supplier_costs, capacity, supply_limit, Cr, Ni, Cr_required, Ni_required, CopperLimit, Cu