import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd
from Pan_6134130_data import get_data_b  # Import from different data (change data names)


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

    return model, x, I, z


# Pass data to the model and solve
def set_data_and_solve(model, x, I, z, demand, holding_costs, supplier_costs,
                       capacity, supply_limit, Cr, Ni, Cr_required, Ni_required):
    """
    Set data into the model and add relevant constraints, then solve it.
    """
    n_products = len(demand)
    n_suppliers = len(supplier_costs)
    n_months = len(demand[0])

    # Set the objective function: minimize inventory holding costs and procurement costs
    objective = gp.quicksum(holding_costs[p] * I[p, t] for p in range(n_products) for t in range(n_months)) + \
                gp.quicksum(supplier_costs[s] * z[p, s, t] for p in range(n_products) for s in range(n_suppliers) for t in range(n_months))
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
        for p in range(n_products):
        # Chrome constraint
            model.addConstr(
                gp.quicksum(Cr[s] * z[p, s, t] for s in range(n_suppliers)) ==
                Cr_required[p] * x[p, t]
            )
            # Nickel constraint
            model.addConstr(
                gp.quicksum(Ni[s] * z[p, s, t] for s in range(n_suppliers)) ==
                Ni_required[p] * x[p, t]
            )
            model.addConstr(
                gp.quicksum(z[p, s, t] for s in range(n_suppliers)) == x[p, t])
        # Balance constraint: the raw material purchased should match production needs



    # Start optimization
    model.optimize()

    # Output results
    if model.status == GRB.OPTIMAL:
        print(f"Optimal objective value: {model.objVal} EURO")

        # Output decision variables
        production_plan = np.zeros((n_products, n_months))
        inventory_plan = np.zeros((n_products, n_months))
        purchase_plan = np.zeros((n_products, n_suppliers, n_months))

        # Extract production plan and inventory
        for p in range(n_products):
            for t in range(n_months):
                production_plan[p, t] = x[p, t].x
                inventory_plan[p, t] = I[p, t].x

        # Extract procurement plan for each supplier and each product
        for p in range(n_products):
            for s in range(n_suppliers):
                for t in range(n_months):
                    purchase_plan[p, s, t] = z[p, s, t].x

        # Use pandas to display production, inventory, and procurement plans
        products = [f'Product {p + 1}' for p in range(n_products)]
        suppliers = [f'Supplier {s + 1}' for s in range(n_suppliers)]
        months = [f'Month {t + 1}' for t in range(n_months)]

        # Create production plan table
        production_df = pd.DataFrame(production_plan, index=products, columns=months)
        inventory_df = pd.DataFrame(inventory_plan, index=products, columns=months)

        # Create procurement plan table for each supplier and product
        purchase_df_dict = {}
        for p in range(n_products):
            purchase_df_dict[products[p]] = pd.DataFrame(purchase_plan[p], index=suppliers, columns=months)

        # Output production and inventory plan
        print("\nProduction Plan (kg):")
        print(production_df.to_string())

        print("\nInventory Plan (kg):")
        print(inventory_df.to_string())

        # Output procurement plan by product and supplier
        for product, purchase_df in purchase_df_dict.items():
            print(f"\nProcurement Plan for {product} (kg):")
            print(purchase_df.to_string())

    else:
        print("No optimal solution found.")


# Main function: Get data, define the model, and solve
if __name__ == "__main__":
    # Get data
    demand, holding_costs, supplier_costs, capacity, supply_limit, Cr, Ni, Cr_required, Ni_required = get_data_b()# Change data name here

    # Define model
    model, x, I, z = define_model(n_products=len(demand), n_suppliers=len(supplier_costs), n_months=len(demand[0]))

    # Solve the model
    set_data_and_solve(model, x, I, z, demand, holding_costs, supplier_costs, capacity, supply_limit, Cr, Ni, Cr_required, Ni_required)
