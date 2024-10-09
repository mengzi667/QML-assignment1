import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd
from b_data import get_data  # Import data from the data file

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

    # z - Procurement matrix (supplier)
    z = model.addMVar((n_suppliers, n_months), lb=0, name="z")

    # Copper-related variables
    Cu = model.addMVar(n_months, lb=0, name="Cu")  # Amount of copper

    # Binary variable for electrolysis decision
    y = model.addMVar(n_months, vtype=GRB.BINARY, name="y")  # Electrolysis activation

    # Amount of copper removed by electrolysis
    removed_copper = model.addMVar(n_months, lb=0, name="removed_copper")

    return model, x, I, z, Cu, y, removed_copper


# Pass data to the model and solve
def set_data_and_solve(model, x, I, z, Cu, y, removed_copper, demand, holding_costs, supplier_costs,
                       capacity, supply_limit, Cr, Ni, Cr_required, Ni_required, CopperLimit):
    n_products = len(demand)
    n_suppliers = len(supplier_costs)
    n_months = len(demand[0])

    # 1. Set the objective function: minimize inventory holding costs, procurement costs, and electrolysis costs
    objective = gp.quicksum(holding_costs[i] * I[i, t] for i in range(n_products) for t in range(n_months)) + \
                gp.quicksum(supplier_costs[j] * z[j, t] for j in range(n_suppliers) for t in range(n_months)) + \
                gp.quicksum(100 * y[t] for t in range(n_months)) + \
                gp.quicksum(5 * removed_copper[t] for t in range(n_months))

    # 2. Set the objective function
    model.setObjective(objective, GRB.MINIMIZE)

    # 3. Demand satisfaction constraints
    for i in range(n_products):
        for t in range(n_months):
            if t == 0:
                model.addConstr(x[i, t] >= demand[i][t] + I[i, t])  # Initial inventory is 0
            else:
                model.addConstr(x[i, t] + I[i, t - 1] == demand[i][t] + I[i, t])

    # 4. Production capacity constraints
    for t in range(n_months):
        model.addConstr(gp.quicksum(x[i, t] for i in range(n_products)) <= capacity[t])

    # 5. Supplier material monthly supply limits
    for j in range(n_suppliers):
        for t in range(n_months):
            model.addConstr(z[j, t] <= supply_limit[j])

    # 6. Chromium and nickel content constraints
    for t in range(n_months):
        # Chromium
        model.addConstr(gp.quicksum(Cr[j] * z[j, t] for j in range(n_suppliers)) ==
                        gp.quicksum(Cr_required[i] * x[i, t] for i in range(n_products)))

        # Nickel
        model.addConstr(gp.quicksum(Ni[j] * z[j, t] for j in range(n_suppliers)) ==
                        gp.quicksum(Ni_required[i] * x[i, t] for i in range(n_products)))

    # 7. Copper content constraint
    for t in range(n_months):
        model.addConstr(Cu[t] <= CopperLimit + removed_copper[t])

    # 8. Copper removal constraint (only if electrolysis is activated)
    for t in range(n_months):
        model.addConstr(removed_copper[t] >= Cu[t] - CopperLimit)
        model.addConstr(removed_copper[t] <= 1000 * y[t])  # Using big M = 1000

    # Start optimization
    model.optimize()

    # Output results
    if model.status == GRB.OPTIMAL:
        print(f"Optimal objective value: {model.objVal} EURO")

        # Output decision variables
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

        # Extract copper-related results
        removed_copper_plan = np.array([removed_copper[t].x for t in range(n_months)])
        electrolysis_activation = np.array([y[t].x for t in range(n_months)])

        # Use pandas to display production, inventory, procurement plans, and electrolysis
        products = ['18/10', '18/8', '18/0']
        suppliers = ['A', 'B', 'C', 'D', 'E']
        months = [f'Month {t + 1}' for t in range(n_months)]

        # Create dataframes for plans
        production_df = pd.DataFrame(production_plan, index=products, columns=months)
        inventory_df = pd.DataFrame(inventory_plan, index=products, columns=months)
        purchase_df = pd.DataFrame(purchase_plan, index=suppliers, columns=months)
        electrolysis_df = pd.DataFrame({'Removed Copper': removed_copper_plan, 'Electrolysis Activated': electrolysis_activation}, index=months)

        print("\nProduction Plan (kg):")
        print(production_df.to_string())

        print("\nInventory Plan (kg):")
        print(inventory_df.to_string())

        print("\nProcurement Plan (kg):")
        print(purchase_df.to_string())

        print("\nElectrolysis Plan (kg):")
        print(electrolysis_df.to_string())

    else:
        print("No optimal solution found.")


# Main function: Get data, define the model, and solve
if __name__ == "__main__":
    # Get data
    demand, holding_costs, supplier_costs, capacity, supply_limit, Cr, Ni, Cu, Cr_required, Ni_required, CopperLimit = get_data()

    # Define the model
    model, x, I, z, Cu, y, removed_copper = define_model(n_products=3, n_suppliers=5, n_months=12)

    # Solve the model
    set_data_and_solve(model, x, I, z, Cu, y, removed_copper, demand, holding_costs, supplier_costs, capacity,
                       supply_limit, Cr, Ni, Cr_required, Ni_required, CopperLimit)
