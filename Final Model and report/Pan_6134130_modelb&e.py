import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd

def get_data_b():
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
    Cr_required = np.array([18, 18, 18])
    Ni_required = np.array([10, 8, 0])

    return demand, holding_costs, supplier_costs, capacity, supply_limit, Cr, Ni, Cr_required, Ni_required

def get_data_e():
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
    CopperLimit = np.full(n_months, 1.8335)# change copper limit here

    return demand, holding_costs, supplier_costs, capacity, supply_limit, Cr, Ni, Cr_required, Ni_required, CopperLimit, Cu

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
    Cu_removed = model.addVars(n_products, n_months, vtype=GRB.CONTINUOUS, name="Cu_removed")

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

        production_plan = np.zeros((n_products, n_months))
        inventory_plan = np.zeros((n_products, n_months))
        purchase_plan = np.zeros((n_products, n_suppliers, n_months))

        for p in range(n_products):
            for t in range(n_months):
                production_plan[p, t] = x[p, t].x
                inventory_plan[p, t] = I[p, t].x

        for p in range(n_products):
            for s in range(n_suppliers):
                for t in range(n_months):
                    purchase_plan[p, s, t] = z[p, s, t].x

        products = [f'Product {p + 1}' for p in range(n_products)]
        suppliers = [f'Supplier {s + 1}' for s in range(n_suppliers)]
        months = [f'Month {t + 1}' for t in range(n_months)]

        production_df = pd.DataFrame(production_plan, index=products, columns=months)
        inventory_df = pd.DataFrame(inventory_plan, index=products, columns=months)

        purchase_df_dict = {}
        for p in range(n_products):
            purchase_df_dict[products[p]] = pd.DataFrame(purchase_plan[p], index=suppliers, columns=months)

        print("\nProduction Plan (kg):")
        print(production_df.to_string())

        print("\nInventory Plan (kg):")
        print(inventory_df.to_string())

        for product, purchase_df in purchase_df_dict.items():
            print(f"\nProcurement Plan for {product} (kg):")
            print(purchase_df.to_string())

    else:
        print("No optimal solution found.")

def set_data_and_solve_with_copper(model, x, I, z, y, Cu_removed, demand, holding_costs, supplier_costs, capacity, supply_limit, Cr, Ni, Cr_required, Ni_required, CopperLimit, Cu):
    n_products = len(demand)
    n_suppliers = len(supplier_costs)
    n_months = len(demand[0])

    objective = gp.quicksum(holding_costs[p] * I[p, t] for p in range(n_products) for t in range(n_months)) + \
                gp.quicksum(supplier_costs[s] * z[p, s, t] for p in range(n_products) for s in range(n_suppliers) for t in range(n_months)) + \
                gp.quicksum(100 * y[t] + 5 * gp.quicksum(Cu_removed[p, t] for p in range(n_products)) for t in range(n_months))

    model.setObjective(objective, GRB.MINIMIZE)

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
            model.addConstr(gp.quicksum(Cu[s] * z[p, s, t] for s in range(n_suppliers)) - Cu_removed[p, t] <= CopperLimit[t] * (x[p, t] - Cu_removed[p, t]))
            model.addConstr(Cu_removed[p, t] <= y[t] * gp.quicksum(Cu[s] * z[p, s, t] for s in range(n_suppliers)))
            model.addConstr(gp.quicksum(z[p, s, t] for s in range(n_suppliers)) == x[p, t])
            model.addConstr(gp.quicksum(Cr[s] * z[p, s, t] for s in range(n_suppliers)) == Cr_required[p] * (x[p, t] - Cu_removed[p, t]))
            model.addConstr(gp.quicksum(Ni[s] * z[p, s, t] for s in range(n_suppliers)) == Ni_required[p] * (x[p, t] - Cu_removed[p, t]))
    model.optimize()

    if model.status == GRB.OPTIMAL:
        print(f"Optimal objective value: {model.objVal} EURO")

        production_plan = np.zeros((n_products, n_months))
        inventory_plan = np.zeros((n_products, n_months))
        purchase_plan = np.zeros((n_products, n_suppliers, n_months))
        electrolysis_plan = np.zeros(n_months)
        copper_removed_plan = np.zeros((n_products, n_months))
        holding_costs_total = 0
        electrolysis_costs_total = 0

        for p in range(n_products):
            for t in range(n_months):
                production_plan[p, t] = x[p, t].x
                inventory_plan[p, t] = I[p, t].x
                holding_costs_total += holding_costs[p] * I[p, t].x

        for p in range(n_products):
            for s in range(n_suppliers):
                for t in range(n_months):
                    purchase_plan[p, s, t] = z[p, s, t].x

        for t in range(n_months):
            electrolysis_plan[t] = y[t].x
            electrolysis_costs_total += 100 * y[t].x
            for p in range(n_products):
                copper_removed_plan[p, t] = Cu_removed[p, t].x

        products = [f'Product {p + 1}' for p in range(n_products)]
        suppliers = [f'Supplier {s + 1}' for s in range(n_suppliers)]
        months = [f'Month {t + 1}' for t in range(n_months)]

        production_df = pd.DataFrame(production_plan, index=products, columns=months)
        inventory_df = pd.DataFrame(inventory_plan, index=products, columns=months)

        purchase_df_dict = {}
        for p in range(n_products):
            purchase_df_dict[products[p]] = pd.DataFrame(purchase_plan[p], index=suppliers, columns=months)

        copper_removed_df = pd.DataFrame(copper_removed_plan, index=products, columns=months)
        electrolysis_df = pd.DataFrame({
            'Electrolysis Used': electrolysis_plan
        }, index=months)

        print("\nProduction Plan (kg):")
        print(production_df.to_string())

        print("\nInventory Plan (kg):")
        print(inventory_df.to_string())

        for product, purchase_df in purchase_df_dict.items():
            print(f"\nProcurement Plan for {product} (kg):")
            print(purchase_df.to_string())

        print("\nElectrolysis Plan:")
        print(electrolysis_df.to_string())

        print("\nCopper Removed Plan (kg):")
        print(copper_removed_df.to_string())

        print(f"\nTotal Holding Costs: {holding_costs_total} EURO")
        print(f"Total Electrolysis Costs: {electrolysis_costs_total} EURO")

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