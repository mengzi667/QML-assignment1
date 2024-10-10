import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd
from e_data import get_data

def define_model(n_products, n_suppliers, n_months, tolerance=1e-6):
    model = gp.Model('StainlessSteelProduction_MatrixForm')
    model.setParam('OptimalityTol', tolerance)
    model.setParam('FeasibilityTol', tolerance)

    x = model.addMVar((n_products, n_months), lb=0, name="x")
    I = model.addMVar((n_products, n_months), lb=0, name="I")
    z = model.addMVar((n_products, n_suppliers, n_months), lb=0, name="z")
    y = model.addMVar(n_months, vtype=GRB.BINARY, name="y")
    e = model.addMVar((n_products, n_months), lb=0, name="e")

    return model, x, I, z, y, e

def set_data_and_solve(model, x, I, z, y, e, demand, holding_costs, supplier_costs,
                       capacity, supply_limit, Cr, Ni, Cu, Cr_required, Ni_required, CopperLimit):
    n_products = len(demand)
    n_suppliers = len(supplier_costs)
    n_months = len(demand[0])

    objective = gp.quicksum(holding_costs[i] * I[i, t] for i in range(n_products) for t in range(n_months)) + \
                gp.quicksum(supplier_costs[j] * z[p, j, t] for p in range(n_products) for j in range(n_suppliers) for t in range(n_months)) + \
                gp.quicksum(100 * y[t] for t in range(n_months)) + \
                gp.quicksum(5 * e[i, t] for i in range(n_products) for t in range(n_months))
    model.setObjective(objective, GRB.MINIMIZE)

    for i in range(n_products):
        for t in range(n_months):
            if t == 0:
                model.addConstr(x[i, t] >= demand[i][t] + I[i, t])
            else:
                model.addConstr(x[i, t] + I[i, t - 1] == demand[i][t] + I[i, t])

    for t in range(n_months):
        model.addConstr(gp.quicksum(x[i, t] - e[i, t] for i in range(n_products)) <= capacity[t])

    for j in range(n_suppliers):
        for t in range(n_months):
            model.addConstr(gp.quicksum(z[p, j, t] for p in range(n_products)) <= supply_limit[j])

    for t in range(n_months):
        model.addConstr(gp.quicksum(Cr[j] * z[p, j, t] for p in range(n_products) for j in range(n_suppliers)) == gp.quicksum(Cr_required[i] * x[i, t] for i in range(n_products)))
        model.addConstr(gp.quicksum(Ni[j] * z[p, j, t] for p in range(n_products) for j in range(n_suppliers)) == gp.quicksum(Ni_required[i] * x[i, t] for i in range(n_products)))

    for t in range(n_months):
        model.addConstr(gp.quicksum(Cu[j] * z[p, j, t] for p in range(n_products) for j in range(n_suppliers)) - gp.quicksum(e[i, t] for i in range(n_products)) <= CopperLimit[t] * gp.quicksum(x[i, t] - e[i, t] for i in range(n_products)))

    M = 1000
    for t in range(n_months):
        for i in range(n_products):
            model.addConstr(e[i, t] <= M * y[t])

    model.optimize()

    if model.status == GRB.OPTIMAL:
        print(f"Optimal objective value: {model.objVal} Euros")

        production_plan = np.zeros((n_products, n_months))
        inventory_plan = np.zeros((n_products, n_months))
        purchase_plan = np.zeros((n_products, n_suppliers, n_months))

        for i in range(n_products):
            for t in range(n_months):
                production_plan[i, t] = x[i, t].x
                inventory_plan[i, t] = I[i, t].x

        for p in range(n_products):
            for j in range(n_suppliers):
                for t in range(n_months):
                    purchase_plan[p, j, t] = z[p, j, t].x

        products = ['18/10', '18/8', '18/0']
        suppliers = ['A', 'B', 'C', 'D', 'E']
        months = [f'Month {t + 1}' for t in range(n_months)]

        production_df = pd.DataFrame(production_plan, index=products, columns=months)
        inventory_df = pd.DataFrame(inventory_plan, index=products, columns=months)
        purchase_df_dict = {products[p]: pd.DataFrame(purchase_plan[p], index=suppliers, columns=months) for p in range(n_products)}

        print("\nProduction Plan (kg):")
        print(production_df.to_string())

        print("\nInventory Plan (kg):")
        print(inventory_df.to_string())

        for product, purchase_df in purchase_df_dict.items():
            print(f"\nProcurement Plan for {product} (kg):")
            print(purchase_df.to_string())

        electrolysis_use = [y[t].x for t in range(n_months)]
        copper_removed = np.zeros((n_products, n_months))
        for i in range(n_products):
            for t in range(n_months):
                copper_removed[i, t] = e[i, t].x

        print("\nElectrolysis Usage:")
        print(pd.DataFrame([electrolysis_use], index=['Usage'], columns=months).to_string())

        print("\nCopper Removed (kg):")
        print(pd.DataFrame(copper_removed, index=products, columns=months).to_string())

    else:
        print("No optimal solution found.")

if __name__ == "__main__":
    demand, holding_costs, supplier_costs, capacity, supply_limit, Cr, Ni, Cr_required, Ni_required, CopperLimit, Cu = get_data()
    model, x, I, z, y, e = define_model(n_products=3, n_suppliers=5, n_months=12)
    set_data_and_solve(model, x, I, z, y, e, demand, holding_costs, supplier_costs, capacity, supply_limit, Cr, Ni, Cu, Cr_required, Ni_required, CopperLimit)