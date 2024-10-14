import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd

# Define the model (three subscript variables), including setting precision
def define_model(n_products, n_suppliers, n_months, include_copper=False, tolerance=1e-6):
    model = gp.Model('Three_Subscript_Model')
    model.setParam('OptimalityTol', tolerance)
    model.setParam('FeasibilityTol', tolerance)

    x = model.addVars(n_products, n_months, vtype=GRB.CONTINUOUS, name="x")
    I = model.addVars(n_products, n_months, vtype=GRB.CONTINUOUS, name="I")
    z = model.addVars(n_products, n_suppliers, n_months, vtype=GRB.CONTINUOUS, name="z")

    if include_copper:
        y = model.addVars(n_months, vtype=GRB.BINARY, name="y")
        Cu_removed = model.addVars(n_products, n_suppliers, n_months, vtype=GRB.CONTINUOUS, name="Cu_removed")
    else:
        y = None
        Cu_removed = None

    return model, x, I, z, y, Cu_removed

# Pass data to the model and solve
def set_data_and_solve(model, x, I, z, y, Cu_removed, demand, holding_costs, supplier_costs,
                       capacity, supply_limit, Cr, Ni, Cr_required, Ni_required, CopperLimit, Cu, include_copper=False):
    n_products = len(demand)
    n_suppliers = len(supplier_costs)
    n_months = len(demand[0])

    objective = gp.quicksum(holding_costs[p] * I[p, t] for p in range(n_products) for t in range(n_months)) + \
                gp.quicksum(supplier_costs[s] * z[p, s, t] for p in range(n_products) for s in range(n_suppliers) for t in range(n_months))

    if include_copper:
        objective += gp.quicksum(100 * y[t] + 5 * gp.quicksum(Cu_removed[p, s, t] for p in range(n_products) for s in range(n_suppliers)) for t in range(n_months))

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
        model.addConstr(
            gp.quicksum(Cr[s] * z[p, s, t] for s in range(n_suppliers) for p in range(n_products)) ==
            gp.quicksum(Cr_required[p] * x[p, t] for p in range(n_products))
        )
        model.addConstr(
            gp.quicksum(Ni[s] * z[p, s, t] for s in range(n_suppliers) for p in range(n_products)) ==
            gp.quicksum(Ni_required[p] * x[p, t] for p in range(n_products))
        )
        model.addConstr(
            gp.quicksum(z[p, s, t] for s in range(n_suppliers) for p in range(n_products)) ==
            gp.quicksum(x[p, t] for p in range(n_products))
        )

    if include_copper:
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

    model.optimize()

    if model.status == GRB.OPTIMAL:
        print(f"Optimal objective value: {model.objVal} EURO")

        holding_cost = sum(holding_costs[p] * I[p, t].X for p in range(n_products) for t in range(n_months))
        electrolysis_cost = sum(100 * y[t].X + 5 * sum(Cu_removed[p, s, t].X for p in range(n_products) for s in range(n_suppliers)) for t in range(n_months))

        print(f"Holding Cost: {holding_cost} EURO")
        print(f"Electrolysis Cost: {electrolysis_cost} EURO")

        production_plan = np.zeros((n_products, n_months))
        inventory_plan = np.zeros((n_products, n_months))
        purchase_plan = np.zeros((n_products, n_suppliers, n_months))
        electrolysis_plan = np.zeros(n_months) if include_copper else None
        copper_removed_plan = np.zeros((n_products, n_suppliers, n_months)) if include_copper else None

        for p in range(n_products):
            for t in range(n_months):
                production_plan[p, t] = x[p, t].x
                inventory_plan[p, t] = I[p, t].x

        for p in range(n_products):
            for s in range(n_suppliers):
                for t in range(n_months):
                    purchase_plan[p, s, t] = z[p, s, t].x

        if include_copper:
            for t in range(n_months):
                electrolysis_plan[t] = y[t].x
                for p in range(n_products):
                    for s in range(n_suppliers):
                        copper_removed_plan[p, s, t] = Cu_removed[p, s, t].x

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

        if include_copper:
            copper_removed_df = pd.DataFrame(copper_removed_plan.reshape(n_products * n_suppliers, n_months),
                                             index=pd.MultiIndex.from_product([products, suppliers]),
                                             columns=months)
            electrolysis_df = pd.DataFrame({
                'Electrolysis Used': electrolysis_plan
            }, index=months)

            print("\nElectrolysis Plan:")
            print(electrolysis_df.to_string())

            print("\nCopper Removed Plan (kg):")
            print(copper_removed_df.to_string())

    else:
        print("No optimal solution found.")

# Main function: Get data, define the model, and solve
if __name__ == "__main__":
    include_copper = True

    if include_copper:
        from e_data import get_data
    else:
        from b_data import get_data

    data = get_data()
    if include_copper:
        demand, holding_costs, supplier_costs, capacity, supply_limit, Cr, Ni, Cr_required, Ni_required, CopperLimit, Cu = data
    else:
        demand, holding_costs, supplier_costs, capacity, supply_limit, Cr, Ni, Cr_required, Ni_required = data
        CopperLimit, Cu = None, None

    model, x, I, z, y, Cu_removed = define_model(n_products=len(demand), n_suppliers=len(supplier_costs), n_months=len(demand[0]), include_copper=include_copper)
    set_data_and_solve(model, x, I, z, y, Cu_removed, demand, holding_costs, supplier_costs, capacity, supply_limit, Cr, Ni, Cr_required, Ni_required, CopperLimit, Cu, include_copper=include_copper)