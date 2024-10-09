import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd
from e_data import get_data


def define_model(n_products, n_suppliers, n_months, tolerance=1e-6):
    """
    Define the optimization model for stainless steel production.

    Parameters:
    n_products (int): Number of products.
    n_suppliers (int): Number of suppliers.
    n_months (int): Number of months in the planning horizon.
    tolerance (float): Tolerance for optimization precision.

    Returns:
    tuple: A tuple containing the model and decision variables (x, I, z, y, e).
    """
    # Create a new Gurobi model
    model = gp.Model('StainlessSteelProduction_MatrixForm')

    # Set optimization precision
    model.setParam('OptimalityTol', tolerance)
    model.setParam('FeasibilityTol', tolerance)

    # Decision variables:
    # x - Production matrix
    x = model.addMVar((n_products, n_months), lb=0, name="x")

    # I - Inventory matrix
    I = model.addMVar((n_products, n_months), lb=0, name="I")

    # z - Procurement matrix
    z = model.addMVar((n_suppliers, n_months), lb=0, name="z")

    # y - Binary variable indicating whether electrolysis is used each month
    y = model.addMVar(n_months, vtype=GRB.BINARY, name="y")

    # e - Amount of copper removed each month for each product
    e = model.addMVar((n_products, n_months), lb=0, name="e")

    return model, x, I, z, y, e


def set_data_and_solve(model, x, I, z, y, e, demand, holding_costs, supplier_costs,
                       capacity, supply_limit, Cr, Ni, Cu, Cr_required, Ni_required, CopperLimit):
    """
    Set data and solve the optimization model.

    Parameters:
    model (gurobipy.Model): The Gurobi model.
    x (gurobipy.MVar): Production matrix.
    I (gurobipy.MVar): Inventory matrix.
    z (gurobipy.MVar): Procurement matrix.
    y (gurobipy.MVar): Binary variable for electrolysis usage.
    e (gurobipy.MVar): Copper removal matrix.
    demand (np.array): Demand for each product in each month.
    holding_costs (np.array): Inventory holding costs for each product.
    supplier_costs (np.array): Cost per kilogram of material from each supplier.
    capacity (np.array): Monthly production capacity.
    supply_limit (np.array): Maximum supply quantity per supplier each month.
    Cr (np.array): Chromium content percentages from each supplier.
    Ni (np.array): Nickel content percentages from each supplier.
    Cu (np.array): Copper content percentages from each supplier.
    Cr_required (np.array): Required Chromium content for each product.
    Ni_required (np.array): Required Nickel content for each product.
    CopperLimit (np.array): Monthly copper limit.

    Returns:
    None
    """
    n_products = len(demand)
    n_suppliers = len(supplier_costs)
    n_months = len(demand[0])

    # 1. Set the objective function: minimize inventory holding costs, procurement costs, and electrolysis costs
    objective = gp.quicksum(holding_costs[i] * I[i, t] for i in range(n_products) for t in range(n_months)) + \
                gp.quicksum(supplier_costs[j] * z[j, t] for j in range(n_suppliers) for t in range(n_months)) + \
                gp.quicksum(100 * y[t] for t in range(n_months)) + \
                gp.quicksum(5 * e[i, t] for i in range(n_products) for t in range(n_months))

    # 2. Set the objective function in the model
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
        model.addConstr(gp.quicksum(x[i, t] - e[i, t] for i in range(n_products)) <= capacity[t])

    # 5. Supplier monthly material supply constraints
    for j in range(n_suppliers):
        for t in range(n_months):
            model.addConstr(z[j, t] <= supply_limit[j])

    # 6. Chromium and Nickel content constraints
    for t in range(n_months):
        # Chromium
        model.addConstr(
            gp.quicksum(Cr[j] * z[j, t] for j in range(n_suppliers)) == gp.quicksum(
                Cr_required[i] * x[i, t] for i in range(n_products))
        )
        # Nickel
        model.addConstr(
            gp.quicksum(Ni[j] * z[j, t] for j in range(n_suppliers)) == gp.quicksum(
                Ni_required[i] * x[i, t] for i in range(n_products))
        )

    # 7. Copper content constraints
    for t in range(n_months):
        model.addConstr(
            gp.quicksum(Cu[j] * z[j, t] for j in range(n_suppliers)) -
            gp.quicksum(e[i, t] for i in range(n_products)) <=
            CopperLimit[t] * gp.quicksum(x[i, t] - e[i, t] for i in range(n_products))
        )

    # 8. Electrolysis usage constraints
    M = 1000  # A sufficiently large number
    for t in range(n_months):
        for i in range(n_products):
            model.addConstr(e[i, t] <= M * y[t])

    # Start optimization
    model.optimize()

    # Output results
    if model.status == GRB.OPTIMAL:
        print(f"Optimal objective value: {model.objVal} Euros")

        # Output decision variables
        production_plan = np.zeros((n_products, n_months))
        inventory_plan = np.zeros((n_products, n_months))
        purchase_plan = np.zeros((n_suppliers, n_months))

        # Extract production and inventory plans
        for i in range(n_products):
            for t in range(n_months):
                production_plan[i, t] = x[i, t].x
                inventory_plan[i, t] = I[i, t].x

        # Extract procurement plan
        for j in range(n_suppliers):
            for t in range(n_months):
                purchase_plan[j, t] = z[j, t].x

        # Use pandas to display production, inventory, and procurement plans
        products = ['18/10', '18/8', '18/0']
        suppliers = ['A', 'B', 'C', 'D', 'E']
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

        # Output electrolysis usage
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


# Main function
if __name__ == "__main__":
    # Get data
    demand, holding_costs, supplier_costs, capacity, supply_limit, Cr, Ni, Cr_required, Ni_required, CopperLimit, Cu = get_data()

    # Define model
    model, x, I, z, y, e = define_model(n_products=3, n_suppliers=5, n_months=12)

    # Solve model
    set_data_and_solve(model, x, I, z, y, e, demand, holding_costs, supplier_costs, capacity,
                       supply_limit, Cr, Ni, Cu, Cr_required, Ni_required, CopperLimit)