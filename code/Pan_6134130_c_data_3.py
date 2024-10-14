import numpy as np


def get_data():
    """
    Generates and returns various data arrays used for supply chain management.

    Returns:
        tuple: A tuple containing the following elements:
            - demand (numpy.ndarray): Monthly demand for each product.
            - holding_costs (numpy.ndarray): Inventory holding costs for each product (Euro/kg).
            - supplier_costs (numpy.ndarray): Supplier cost per kilogram of material (Euro).
            - capacity (numpy.ndarray): Monthly production capacity limits.
            - supply_limit (numpy.ndarray): Maximum supply quantity per supplier each month.
            - Cr (numpy.ndarray): Chromium percentages in materials provided by suppliers.
            - Ni (numpy.ndarray): Nickel percentages in materials provided by suppliers.
            - Cr_required (numpy.ndarray): Required Chromium content for each product.
            - Ni_required (numpy.ndarray): Required Nickel content for each product.
    """
    # Number of months, products, and suppliers
    n_months = 12
    n_products = 3
    n_suppliers = 5

    # Monthly demand for each product
    demand = np.array([
        [10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 18/10
        [10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 18/8
        [10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 18/0
    ])

    # Inventory holding costs for each product (Euro/kg)
    holding_costs = np.array([20, 10, 5])

    # Supplier cost per kilogram of material (Euro)
    supplier_costs = np.array([5, 10, 9, 7, 8.5])

    # Monthly production capacity limits
    capacity = np.full(n_months, 100)

    # Maximum supply quantity per supplier each month
    supply_limit = np.array([90, 30, 50, 70, 20])

    # Chromium and Nickel percentages in materials provided by suppliers
    Cr = np.array([18, 25, 15, 14, 0])
    Ni = np.array([0, 15, 10, 16, 10])

    # Required Chromium and Nickel content for each product
    Cr_required = np.array([18, 18, 18])
    Ni_required = np.array([10, 8, 0])

    return demand, holding_costs, supplier_costs, capacity, supply_limit, Cr, Ni, Cr_required, Ni_required
