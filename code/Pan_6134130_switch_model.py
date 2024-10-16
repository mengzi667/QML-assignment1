import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd
from Pan_6134130_data import get_data_b, get_data_e

# Define the model (three subscript variables), including setting precision
def define_model(n_products, n_suppliers, n_months, tolerance=1e-6):
    model = gp.Model('Three_Subscript_Model')
    model.setParam('OptimalityTol', tolerance)
    model.setParam('FeasibilityTol', tolerance)

    x = model.addVars(n_products, n_months, vtype=GRB.CONTINUOUS, name="x")
    I = model.addVars(n_products, n_months, vtype=GRB.CONTINUOUS, name="I")
    z = model.addVars(n_products, n_suppliers, n_months, vtype=GRB.CONTINUOUS, name="z")

    return model, x, I, z

def define_model_with_copper(n_products, n_suppliers, n_months, tolerance=1e-6):
    model = gp.Model('Three_Subscript_Model_With_Copper')
    model.setParam('OptimalityTol', tolerance)
    model.setParam('FeasibilityTol', tolerance)

    x = model.addVars(n_products, n_months, vtype=GRB.CONTINUOUS, name="x")
    I = model.addVars(n_products, n_months, vtype=GRB.CONTINUOUS, name="I")
    z = model.addVars(n_products, n_suppliers, n_months, vtype=GRB.CONTINUOUS, name="z")
    y = model.addVars(n_months, vtype=GRB.BINARY, name="y")
    Cu_removed = model.addVars(n_products, n_suppliers, n_months, vtype=GRB.CONTINUOUS, name="Cu_removed")

    return model, x, I, z, y, Cu_removed

def set_data_and_solve(model, x, I, z, demand, holding_costs, supplier_costs, capacity, supply_limit, Cr, Ni, Cr_required, Ni_required):
    n_products = len(demand)
    n_suppliers = len(supplier_costs)
    n_months = len(demand[0])

    objective = gp.quicksum(holding_costs[p] * I[p, t] for p in range(n_products) for t in range(n_months)) + \
                gp.quicksum(supplier_costs[s] * z[p, s, t] for p in range(n_products) for s in range(n_suppliers) for t in range(n_months))
    model.setObjective(objective, GRB.MINIMIZE)

    for p in range(n_products):
        for t in range(n_months):
            if t == 0:
                model.addConstr(x[p, t] == demand[p][t] + I[p, t])
            else:
                model.addConstr(x[p, t] + I[p, t - 1] == demand[p][t] + I[p, t])

    for t in range(n_months):
        model.addConstr(gp.quicksum(x[p, t] for p in range(n_products)) <= capacity[t])

    for s in range(n_suppliers):
        for t in range(n_months):
            model.addConstr(gp.quicksum(z[p, s, t] for p in range(n_products)) <= supply_limit[s])

    for t in range(n_months):
        for p in range(n_products):
            model.addConstr(gp.quicksum(Cr[s] * z[p, s, t] for s in range(n_suppliers)) == Cr_required[p] * x[p, t])
            model.addConstr(gp.quicksum(Ni[s] * z[p, s, t] for s in range(n_suppliers)) == Ni_required[p] * x[p, t])
            model.addConstr(gp.quicksum(z[p, s, t] for s in range(n_suppliers)) == x[p, t])

    model.optimize()

    if model.status == GRB.OPTIMAL:
        print(f"Optimal objective value: {model.objVal} EURO")
    else:
        print("No optimal solution found.")

def set_data_and_solve_with_copper(model, x, I, z, y, Cu_removed, demand, holding_costs, supplier_costs, capacity, supply_limit, Cr, Ni, Cr_required, Ni_required, CopperLimit, Cu):
    n_products = len(demand)
    n_suppliers = len(supplier_costs)
    n_months = len(demand[0])

    objective = gp.quicksum(holding_costs[p] * I[p, t] for p in range(n_products) for t in range(n_months)) + \
                gp.quicksum(supplier_costs[s] * z[p, s, t] for p in range(n_products) for s in range(n_suppliers) for t in range(n_months)) + \
                gp.quicksum(100 * y[t] + 5 * gp.quicksum(Cu_removed[p, s, t] for p in range(n_products) for s in range(n_suppliers)) for t in range(n_months))

    model.setObjective(objective, GRB.MINIMIZE)

    for p in range(n_products):
        for t in range(n_months):
            if t == 0:
                model.addConstr(x[p, t] == demand[p][t] + I[p, t])
            else:
                model.addConstr(x[p, t] + I[p, t - 1] == demand[p][t] + I[p, t])

    for t in range(n_months):
        model.addConstr(gp.quicksum(x[p, t] for p in range(n_products)) <= capacity[t])

    for s in range(n_suppliers):
        for t in range(n_months):
            model.addConstr(gp.quicksum(z[p, s, t] for p in range(n_products)) <= supply_limit[s])

    for t in range(n_months):
        model.addConstr(gp.quicksum(Cr[s] * z[p, s, t] for s in range(n_suppliers) for p in range(n_products)) == gp.quicksum(Cr_required[p] * x[p, t] for p in range(n_products)))
        model.addConstr(gp.quicksum(Ni[s] * z[p, s, t] for s in range(n_suppliers) for p in range(n_products)) == gp.quicksum(Ni_required[p] * x[p, t] for p in range(n_products)))
        model.addConstr(gp.quicksum(z[p, s, t] for s in range(n_suppliers) for p in range(n_products)) == gp.quicksum(x[p, t] for p in range(n_products)))

    for t in range(n_months):
        for p in range(n_products):
            model.addConstr(gp.quicksum(Cu[s] * z[p, s, t] for s in range(n_suppliers)) - gp.quicksum(Cu_removed[p, s, t] for s in range(n_suppliers)) <= CopperLimit[t] * x[p, t])
            for s in range(n_suppliers):
                model.addConstr(Cu_removed[p, s, t] <= Cu[s] * z[p, s, t])
                model.addConstr(Cu_removed[p, s, t] <= y[t] * Cu[s] * z[p, s, t])

    model.optimize()

    if model.status == GRB.OPTIMAL:
        print(f"Optimal objective value: {model.objVal} EURO")
    else:
        print("No optimal solution found.")

if __name__ == "__main__":
    model_choice = input("Enter 'b' to run the basic model or 'e' to run the model with copper: ").strip().lower()

    if model_choice == 'b':
        demand, holding_costs, supplier_costs, capacity, supply_limit, Cr, Ni, Cr_required, Ni_required = get_data_b()
        model, x, I, z = define_model(n_products=len(demand), n_suppliers=len(supplier_costs), n_months=len(demand[0]))
        set_data_and_solve(model, x, I, z, demand, holding_costs, supplier_costs, capacity, supply_limit, Cr, Ni, Cr_required, Ni_required)
    elif model_choice == 'e':
        demand, holding_costs, supplier_costs, capacity, supply_limit, Cr, Ni, Cr_required, Ni_required, CopperLimit, Cu = get_data_e()
        model, x, I, z, y, Cu_removed = define_model_with_copper(n_products=len(demand), n_suppliers=len(supplier_costs), n_months=len(demand[0]))
        set_data_and_solve_with_copper(model, x, I, z, y, Cu_removed, demand, holding_costs, supplier_costs, capacity, supply_limit, Cr, Ni, Cr_required, Ni_required, CopperLimit, Cu)
    else:
        print("Invalid input. Please enter 'b' or 'e'.")