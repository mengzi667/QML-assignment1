import numpy as np

def get_data():
    """
    Generates and returns data related to monthly product demands, costs, capacities, and material requirements.

    Returns:
        tuple: A tuple containing the following elements:
            - demand (numpy.ndarray): Monthly demand for each product.
            - holding_costs (numpy.ndarray): Inventory holding costs for each product (in euros per kilogram).
            - supplier_costs (numpy.ndarray): Material costs from each supplier (in euros per kilogram).
            - capacity (numpy.ndarray): Monthly production capacity limits.
            - supply_limit (numpy.ndarray): Maximum monthly supply from each supplier.
            - Cr (numpy.ndarray): Chromium percentage in materials provided by each supplier.
            - Ni (numpy.ndarray): Nickel percentage in materials provided by each supplier.
            - Cr_required (numpy.ndarray): Required chromium content for each product.
            - Ni_required (numpy.ndarray): Required nickel content for each product.
            - CopperLimit (numpy.ndarray): Monthly copper limit.
            - Cu (numpy.ndarray): Copper percentage in materials provided by each supplier.
    """
    n_months, n_products, n_suppliers = 12, 3, 5

    demand = np.array([
        [25, 25, 0, 0, 0, 50, 12, 0, 10, 10, 45, 99],
        [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
        [5, 20, 80, 25, 50, 125, 150, 80, 40, 35, 3, 100]
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
    CopperLimit = np.full(n_months, 2.99)

    return demand, holding_costs, supplier_costs, capacity, supply_limit, Cr, Ni, Cr_required, Ni_required, CopperLimit, Cu