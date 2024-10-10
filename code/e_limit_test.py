import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd
from e_data import get_data
from e_model_test import define_model, set_data_and_solve

def experiment_with_copper_limit():
    # Get data, excluding CopperLimit
    demand, holding_costs, supplier_costs, capacity, supply_limit, Cr, Ni, Cr_required, Ni_required, _, Cu = get_data()

    # Define model
    model, x, I, z, y, e = define_model(n_products=3, n_suppliers=5, n_months=12)

    # Record results for different CopperLimit values
    results = []

    # Experiment with different CopperLimit values
    for limit in np.arange(1.31, 1.41, 0.01):
        # Define CopperLimit here
        CopperLimit = np.full(12, limit)

        # Solve model
        set_data_and_solve(model, x, I, z, y, e, demand, holding_costs, supplier_costs, capacity,
                           supply_limit, Cr, Ni, Cu, Cr_required, Ni_required, CopperLimit)

        # Record results
        if model.status == GRB.OPTIMAL:
            electrolysis_cost = sum(100 * y[t].x for t in range(12))
            holding_cost = sum(holding_costs[i] * I[i, t].x for i in range(3) for t in range(12))
            procurement_cost = sum(supplier_costs[s] * z[p, s, t].x for p in range(3) for s in range(5) for t in range(12))
            total_cost = electrolysis_cost + holding_cost + procurement_cost
            results.append((limit, total_cost))

    # Display results
    results_df = pd.DataFrame(results, columns=['CopperLimit', 'TotalCost'])
    print(results_df)

if __name__ == "__main__":
    experiment_with_copper_limit()