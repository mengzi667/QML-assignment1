import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd
from Pan_6134130_data import get_data_e  # Import data from a different data file

# Define the model (three subscript variables), including setting precision
def define_model(n_products, n_suppliers, n_months, tolerance=1e-6):
    """
    Define a custom optimization model with decision variables for production, inventory,
    and procurement using three subscript indices: product, supplier, and time.
    """
    # Create a  new Gurobi model
    model = gp.Model('Three_Subscript_Model')

    # Set solution precision
    model.setParam('OptimalityTol', tolerance)
    model.setParam('FeasibilityTol', tolerance)

    # Decision variables (with three subscripts):
    # x[p, t] - Production of product p in month t
    x = model.addVars(n_products, n_months, vtype=GRB.CONTINUOUS, name="x")

    # I[p, t] - Inventory of product p at the end of month t
    I = model.addVars(n_products, n_months, vtype=GRB.CONTINUOUS, name="I")

    # z[p, s, t] - Quantity of raw material purchased from supplier s for product p in month t
    z = model.addVars(n_products, n_suppliers, n_months, vtype=GRB.CONTINUOUS, name="z")

    # y[t] - Binary variable indicating whether electrolysis is used in month t
    y = model.addVars(n_months, vtype=GRB.BINARY, name="y")

    # Cu_removed[p, s, t] - Amount of copper removed by electrolysis for product p from supplier s in month t
    Cu_removed = model.addVars(n_products, n_suppliers, n_months, vtype=GRB.CONTINUOUS, name="Cu_removed")

    return model, x, I, z, y, Cu_removed

# Pass data to the model and solve
def set_data_and_solve(model, x, I, z, y, Cu_removed, demand, holding_costs, supplier_costs,
                       capacity, supply_limit, Cr, Ni, Cr_required, Ni_required, CopperLimit, Cu):
    """
    Set data into the model and add relevant constraints, then solve it.
    """
    n_products = len(demand)
    n_suppliers = len(supplier_costs)
    n_months = len(demand[0])

    # Set the objective function: minimize inventory holding costs, procurement costs, and electrolysis costs
    objective = gp.quicksum(holding_costs[p] * I[p, t] for p in range(n_products) for t in range(n_months)) + \
                gp.quicksum(supplier_costs[s] * z[p, s, t] for p in range(n_products) for s in range(n_suppliers) for t in range(n_months)) + \
                gp.quicksum(100 * y[t] + 5 * gp.quicksum(Cu_removed[p, s, t] for p in range(n_products) for s in range(n_suppliers)) for t in range(n_months))
    model.setObjective(objective, GRB.MINIMIZE)

    # Demand satisfaction constraints
    for p in range(n_products):
        for t in range(n_months):
            if t == 0:
                model.addConstr(x[p, t] == demand[p][t] + I[p, t])  # Initial inventory is 0
            else:
                model.addConstr(x[p, t] + I[p, t - 1] == demand[p][t] + I[p, t])

    # Production capacity constraints
    for t in range(n_months):
        model.addConstr(gp.quicksum(x[p, t] for p in range(n_products)) <= capacity[t])

    # Supplier material monthly supply limits
    for s in range(n_suppliers):
        for t in range(n_months):
            model.addConstr(gp.quicksum(z[p, s, t] for p in range(n_products)) <= supply_limit[s])

    # Chrome and nickel content constraints
    for t in range(n_months):
        # Chrome constraint
        model.addConstr(
            gp.quicksum(Cr[s] * z[p, s, t] for s in range(n_suppliers) for p in range(n_products)) ==
            gp.quicksum(Cr_required[p] * x[p, t] for p in range(n_products))
        )
        # Nickel constraint
        model.addConstr(
            gp.quicksum(Ni[s] * z[p, s, t] for s in range(n_suppliers) for p in range(n_products)) ==
            gp.quicksum(Ni_required[p] * x[p, t] for p in range(n_products))
        )
        # Balance constraint: the raw material purchased should match production needs
        model.addConstr(
            gp.quicksum(z[p, s, t] for s in range(n_suppliers) for p in range(n_products)) ==
            gp.quicksum(x[p, t] for p in range(n_products))
        )

    # Copper content constraint
    for t in range(n_months):
        for p in range(n_products):
            model.addConstr(
                gp.quicksum(Cu[s] * z[p, s, t] for s in range(n_suppliers)) - gp.quicksum(Cu_removed[p, s, t] for s in range(n_suppliers)) <= CopperLimit[t] * x[p, t]
            )
            for s in range(n_suppliers):
                model.addConstr(
                    Cu_removed[p, s, t] <= Cu[s] * z[p, s, t]
                )
                model.addConstr(
                    Cu_removed[p, s, t] <= y[t] * Cu[s] * z[p, s, t]
                )

    # Start optimization
    model.optimize()

    # Output results
    if model.status == GRB.OPTIMAL:
        return model.objVal, True
    else:
        return None, False

def find_min_copper_limit(demand, holding_costs, supplier_costs, capacity, supply_limit, Cr, Ni, Cr_required, Ni_required, Cu, original_obj_val):
    """
    Find the minimum CopperLimit that does not increase the original optimal value.
    """
    # Initial range for CopperLimit
    low, high = 0, 5  # Adjust the range as needed
    best_limit = high

    while high - low > 1e-6:  # Precision threshold
        mid = (low + high) / 2
        obj_val, feasible = set_data_and_solve(*define_model(len(demand), len(supplier_costs), len(demand[0])), demand, holding_costs, supplier_costs, capacity, supply_limit, Cr, Ni, Cr_required, Ni_required, [mid]*len(demand[0]), Cu)
        if feasible and obj_val <= original_obj_val:
            best_limit = mid
            high = mid
        else:
            low = mid

    return best_limit

# Main function: Get data, define the model, and solve
if __name__ == "__main__":
    # Get data
    demand, holding_costs, supplier_costs, capacity, supply_limit, Cr, Ni, Cr_required, Ni_required, CopperLimit, Cu = get_data_e()

    # Original optimal value
    original_obj_val = 9646.776415571283

    # Find the minimum CopperLimit
    min_copper_limit = find_min_copper_limit(demand, holding_costs, supplier_costs, capacity, supply_limit, Cr, Ni, Cr_required, Ni_required, Cu, original_obj_val)
    print(f"Minimum CopperLimit that does not increase the original optimal value: {min_copper_limit}")