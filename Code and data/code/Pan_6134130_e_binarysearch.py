import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd


def get_data():
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
    return demand, holding_costs, supplier_costs, capacity, supply_limit, Cr, Ni, Cr_required, Ni_required, Cu


def solve_with_copper_limit(copper_limit, demand, holding_costs, supplier_costs, capacity,
                            supply_limit, Cr, Ni, Cr_required, Ni_required, Cu, base_cost):
    n_products = len(demand)
    n_suppliers = len(supplier_costs)
    n_months = len(demand[0])

    model = gp.Model('Copper_Limit_Model')
    model.setParam('OutputFlag', 0)  # Suppress output

    # Variables
    x = model.addVars(n_products, n_months, vtype=GRB.CONTINUOUS, name="x")
    I = model.addVars(n_products, n_months, vtype=GRB.CONTINUOUS, name="I")
    z = model.addVars(n_products, n_suppliers, n_months, vtype=GRB.CONTINUOUS, name="z")
    y = model.addVars(n_months, vtype=GRB.BINARY, name="y")
    Cu_removed = model.addVars(n_products, n_months, vtype=GRB.CONTINUOUS, name="Cu_removed")

    # Objective
    objective = gp.quicksum(holding_costs[p] * I[p, t] for p in range(n_products) for t in range(n_months)) + \
                gp.quicksum(
                    supplier_costs[s] * z[p, s, t] for p in range(n_products) for s in range(n_suppliers) for t in
                    range(n_months)) + \
                gp.quicksum(
                    100 * y[t] + 5 * gp.quicksum(Cu_removed[p, t] for p in range(n_products)) for t in range(n_months))
    model.setObjective(objective, GRB.MINIMIZE)

    # Constraints
    # Add cost limit constraint
    model.addConstr(objective <= base_cost)

    # Add all other constraints
    for p in range(n_products):
        for t in range(n_months):
            if t == 0:
                model.addConstr((x[p, t] - Cu_removed[p, t]) == demand[p][t] + I[p, t])
            else:
                model.addConstr((x[p, t] - Cu_removed[p, t]) + I[p, t - 1] == demand[p][t] + I[p, t])

    for t in range(n_months):
        model.addConstr(gp.quicksum(x[p, t] for p in range(n_products)) <= capacity[t])

    for s in range(n_suppliers):
        for t in range(n_months):
            model.addConstr(gp.quicksum(z[p, s, t] for p in range(n_products)) <= supply_limit[s])

    for t in range(n_months):
        for p in range(n_products):
            model.addConstr(gp.quicksum(Cu[s] / 100 * z[p, s, t] for s in range(n_suppliers)) - Cu_removed[
                p, t] <= copper_limit / 100 * (x[p, t] - Cu_removed[p, t]))
            model.addConstr(Cu_removed[p, t] <= y[t] * 10000)
            model.addConstr(Cu_removed[p, t] <= gp.quicksum(Cu[s] / 100 * z[p, s, t] for s in range(n_suppliers)))
            model.addConstr(gp.quicksum(z[p, s, t] for s in range(n_suppliers)) == x[p, t])
            model.addConstr(gp.quicksum(Cr[s] * z[p, s, t] for s in range(n_suppliers)) == Cr_required[p] * (
                        x[p, t] - Cu_removed[p, t]))
            model.addConstr(gp.quicksum(Ni[s] * z[p, s, t] for s in range(n_suppliers)) == Ni_required[p] * (
                        x[p, t] - Cu_removed[p, t]))

    model.optimize()

    return model.status == GRB.OPTIMAL


def find_minimum_copper_limit():
    # Get data
    data = get_data()
    demand, holding_costs, supplier_costs, capacity, supply_limit, Cr, Ni, Cr_required, Ni_required, Cu = data

    # First solve with a very high copper limit to get base cost
    test_model = gp.Model('Test_Model')
    test_model.setParam('OutputFlag', 0)
    n_products, n_suppliers, n_months = len(demand), len(supplier_costs), len(demand[0])

    x = test_model.addVars(n_products, n_months, vtype=GRB.CONTINUOUS)
    I = test_model.addVars(n_products, n_months, vtype=GRB.CONTINUOUS)
    z = test_model.addVars(n_products, n_suppliers, n_months, vtype=GRB.CONTINUOUS)
    y = test_model.addVars(n_months, vtype=GRB.BINARY)
    Cu_removed = test_model.addVars(n_products, n_months, vtype=GRB.CONTINUOUS)

    objective = gp.quicksum(holding_costs[p] * I[p, t] for p in range(n_products) for t in range(n_months)) + \
                gp.quicksum(
                    supplier_costs[s] * z[p, s, t] for p in range(n_products) for s in range(n_suppliers) for t in
                    range(n_months)) + \
                gp.quicksum(
                    100 * y[t] + 5 * gp.quicksum(Cu_removed[p, t] for p in range(n_products)) for t in range(n_months))
    test_model.setObjective(objective, GRB.MINIMIZE)

    # Add all constraints except copper limit
    for p in range(n_products):
        for t in range(n_months):
            if t == 0:
                test_model.addConstr((x[p, t] - Cu_removed[p, t]) == demand[p][t] + I[p, t])
            else:
                test_model.addConstr((x[p, t] - Cu_removed[p, t]) + I[p, t - 1] == demand[p][t] + I[p, t])

    for t in range(n_months):
        test_model.addConstr(gp.quicksum(x[p, t] for p in range(n_products)) <= capacity[t])

    for s in range(n_suppliers):
        for t in range(n_months):
            test_model.addConstr(gp.quicksum(z[p, s, t] for p in range(n_products)) <= supply_limit[s])

    for t in range(n_months):
        for p in range(n_products):
            test_model.addConstr(Cu_removed[p, t] <= y[t] * 10000)
            test_model.addConstr(Cu_removed[p, t] <= gp.quicksum(Cu[s] / 100 * z[p, s, t] for s in range(n_suppliers)))
            test_model.addConstr(gp.quicksum(z[p, s, t] for s in range(n_suppliers)) == x[p, t])
            test_model.addConstr(gp.quicksum(Cr[s] * z[p, s, t] for s in range(n_suppliers)) == Cr_required[p] * (
                        x[p, t] - Cu_removed[p, t]))
            test_model.addConstr(gp.quicksum(Ni[s] * z[p, s, t] for s in range(n_suppliers)) == Ni_required[p] * (
                        x[p, t] - Cu_removed[p, t]))

    test_model.optimize()
    base_cost = test_model.objVal

    # Binary search for minimum copper limit
    left, right = 0, 100  # Copper limit percentage range
    min_copper_limit = right
    tolerance = 0.00001  # Tolerance for binary search

    while right - left > tolerance:
        mid = (left + right) / 2
        if solve_with_copper_limit(mid, *data, base_cost):
            min_copper_limit = mid
            right = mid
        else:
            left = mid

    return min_copper_limit, base_cost


if __name__ == "__main__":
    min_copper_limit, base_cost = find_minimum_copper_limit()
    print(f"Minimum required copper limit: {min_copper_limit:.15f}%")
    print(f"Base cost: {base_cost:.8f} EURO")