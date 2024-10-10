import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd
from c_data_4 import get_data  # import data from different data file(jsut change the file name)


# Define the model (using MVar, matrix form), including setting O(ɛ)-precision
def define_model(n_products, n_suppliers, n_months, tolerance=1e-6):
    # Create a new Gurobi model
    model = gp.Model('StainlessSteelProduction_MatrixForm')

    # Set solution precision
    model.setParam('OptimalityTol', tolerance)
    model.setParam('FeasibilityTol', tolerance)

    # Decision variables:
    # x - Production matrix
    x = model.addMVar((n_products, n_months), lb=0, name="x")

    # I - Inventory matrix
    I = model.addMVar((n_products, n_months), lb=0, name="I")

    # z - Procurement matrix
    z = model.addMVar((n_suppliers, n_months), lb=0, name="z")

    return model, x, I, z


# Pass data to the model and solve
def set_data_and_solve(model, x, I, z, demand, holding_costs, supplier_costs,
                       capacity, supply_limit, Cr, Ni, Cr_required, Ni_required):
    n_products = len(demand)
    n_suppliers = len(supplier_costs)
    n_months = len(demand[0])

    # Set the objective function: minimize inventory holding costs and procurement costs
    objective = gp.quicksum(holding_costs[i] * I[i, t] for i in range(n_products) for t in range(n_months)) + \
                gp.quicksum(supplier_costs[j] * z[j, t] for j in range(n_suppliers) for t in range(n_months))
    model.setObjective(objective, GRB.MINIMIZE)

    # Demand satisfaction constraints
    for i in range(n_products):
        for t in range(n_months):
            if t == 0:
                model.addConstr(x[i, t] == demand[i][t] + I[i, t])  # Initial inventory is 0
            else:
                model.addConstr(x[i, t] + I[i, t - 1] == demand[i][t] + I[i, t])

    # Production capacity constraints
    for t in range(n_months):
        model.addConstr(gp.quicksum(x[i, t] for i in range(n_products)) <= capacity[t])

    # Supplier material monthly supply limits
    for j in range(n_suppliers):
        for t in range(n_months):
            model.addConstr(z[j, t] <= supply_limit[j])

    # CHROME and nickel constraints
    for t in range(n_months):
        # Chrome
        model.addConstr(
            gp.quicksum(Cr[j] * z[j, t] for j in range(n_suppliers)) == gp.quicksum(
                Cr_required[i] * x[i, t] for i in range(n_products))
        )
        # NICKEL
        model.addConstr(
            gp.quicksum(Ni[j] * z[j, t] for j in range(n_suppliers)) == gp.quicksum(
                Ni_required[i] * x[i, t] for i in range(n_products))
        )
        # Balance constraint
        model.addConstr(
            gp.quicksum(z[j, t] for j in range(n_suppliers)) == gp.quicksum(
                x[i, t] for i in range(n_products))
        )

    # Start optimization
    model.optimize()

    # Output results
    if model.status == GRB.OPTIMAL:
        print(f"Optimal objective value: {model.objVal} EURO")

        # output Decision variables
        production_plan = np.zeros((n_products, n_months))
        inventory_plan = np.zeros((n_products, n_months))
        purchase_plan = np.zeros((n_suppliers, n_months))

        # Extract production plan and inventory
        for i in range(n_products):
            for t in range(n_months):
                production_plan[i, t] = x[i, t].x
                inventory_plan[i, t] = I[i, t].x

        # Extract procurement plan
        for j in range(n_suppliers):
            for t in range(n_months):
                purchase_plan[j, t] = z[j, t].x

        # Use pandas to display production, inventory, and procurement plans
        products = [f'Product {i + 1}' for i in range(n_products)]
        suppliers = [f'Supplier {j + 1}' for j in range(n_suppliers)]
        months = [f'Month {t + 1}' for t in range(n_months)]

        # Create production plan table
        production_df = pd.DataFrame(production_plan, index=products, columns=months)
        inventory_df = pd.DataFrame(inventory_plan, index=products, columns=months)
        purchase_df = pd.DataFrame(purchase_plan, index=suppliers, columns=months)

        print("\nProduction Plan (kg):")
        print(production_df.to_string())

        print("\nInventory Plan (kg):")
        print(inventory_df.to_string())

        print("\nProcurement Plan (kg):")
        print(purchase_df.to_string())

    else:
        print("No optimal solution found.")


# Main function: Get data, define the model, and solve
if __name__ == "__main__":
    # get data
    demand, holding_costs, supplier_costs, capacity, supply_limit, Cr, Ni, Cr_required, Ni_required = get_data()

    # define model
    model, x, I, z = define_model(n_products=3, n_suppliers=5, n_months=12)

    # solve the model
    set_data_and_solve(model, x, I, z, demand, holding_costs, supplier_costs, capacity,
                       supply_limit, Cr, Ni, Cr_required, Ni_required)
