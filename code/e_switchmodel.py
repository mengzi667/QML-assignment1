import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd

# Define the model (three subscript variables), including setting precision
def define_model(n_products, n_suppliers, n_months, include_copper=False, tolerance=1e-6):
    """
    Define a custom optimization model with decision variables for production, inventory,
    and procurement using three subscript indices: product, supplier, and time.
    """
    # Create a new Gurobi model
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

    if include_copper:
        # y[t] - Binary variable indicating whether electrolysis is used in month t
        y = model.addVars(n_months, vtype=GRB.BINARY, name="y")

        # Cu_removed[p, s, t] - Amount of copper removed by electrolysis for product p from supplier s in month t
        Cu_removed = model.addVars(n_products, n_suppliers, n_months, vtype=GRB.CONTINUOUS, name="Cu_removed")
    else:
        y = None
        Cu_removed = None

    return model, x, I, z, y, Cu_removed

# Pass data to the model and solve
def set_data_and_solve(model, x, I, z, y, Cu_removed, demand, holding_costs, supplier_costs,
                       capacity, supply_limit, Cr, Ni, Cr_required, Ni_required, CopperLimit, Cu, include_copper=False):
    """
    Set data into the model and add relevant constraints, then solve it.
    """
    n_products = len(demand)
    n_suppliers = len(supplier_costs)
    n_months = len(demand[0])

    # Set the objective function: minimize inventory holding costs and procurement costs
    objective = gp.quicksum(holding_costs[p] * I[p, t] for p in range(n_products) for t in range(n_months)) + \
                gp.quicksum(supplier_costs[s] * z[p, s, t] for p in range(n_products) for s in range(n_suppliers) for t in range(n_months))

    if include_copper:
        objective += gp.quicksum(100 * y[t] + 5 * gp.quicksum(Cu_removed[p, s, t] for p in range(n_products) for s in range(n_suppliers)) for t in range(n_months))

    model.setObjective(objective, GRB.MINIMIZE)

    # Demand satisfaction constraints
    for p in range(n_products):
        for t in range(n_months):
            if t == 0:
                model.addConstr(x[p, t] + I[p, t] == demand[p][t])
            else:
                model.addConstr(x[p, t] + I[p, t] == demand[p][t] + I[p, t-1])

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

    if include_copper:
        # Copper content constraint
        for t in range(n_months):
            for p in range(n_products):
                model.addConstr(
                    gp.quicksum(Cu[s] * z[p, s, t] for s in range(n_suppliers)) - Cu_removed[p, s, t] <= CopperLimit[t] * y[t]
                )

    # Start optimization
    model.optimize()

    # Output results
    if model.status == GRB.OPTIMAL:
        print(f"Optimal objective value: {model.objVal} EURO")

        # Calculate and print electrolysis costs and holding costs
        if include_copper:
            electrolysis_costs = sum(100 * y[t].x + 5 * sum(Cu_removed[p, s, t].x for p in range(n_products) for s in range(n_suppliers)) for t in range(n_months))
            print(f"Electrolysis costs: {electrolysis_costs} EURO")

        total_holding_costs = sum(holding_costs[p] * I[p, t].x for p in range(n_products) for t in range(n_months))
        print(f"Holding costs: {total_holding_costs} EURO")

        # Output decision variables
        production_plan = np.zeros((n_products, n_months))
        inventory_plan = np.zeros((n_products, n_months))
        purchase_plan = np.zeros((n_products, n_suppliers, n_months))
        electrolysis_plan = np.zeros(n_months) if include_copper else None
        copper_removed_plan = np.zeros((n_products, n_suppliers, n_months)) if include_copper else None

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

        if include_copper:
            # Extract electrolysis plan and copper removed
            for t in range(n_months):
                electrolysis_plan[t] = y[t].x
                for p in range(n_products):
                    for s in range(n_suppliers):
                        copper_removed_plan[p, s, t] = Cu_removed[p, s, t].x

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

        if include_copper:
            # Output electrolysis plan and copper removed
            copper_removed_df = pd.DataFrame(copper_removed_plan.reshape(n_products * n_suppliers, n_months),
                                             index=pd.MultiIndex.from_product([products, suppliers], names=['Product', 'Supplier']),
                                             columns=months)
            electrolysis_df = pd.DataFrame({
                'Electrolysis': electrolysis_plan
            }, index=months)

            print("\nElectrolysis Plan:")
            print(electrolysis_df.to_string())

            print("\nCopper Removed Plan (kg):")
            print(copper_removed_df.to_string())

    else:
        print("No optimal solution found.")

# Main function: Get data, define the model, and solve
if __name__ == "__main__":
    # Switch to control whether to include copper
    include_copper = True  # Set to False to exclude copper

    if include_copper:
        from e_data import get_data
    else:
        from b_data import get_data

    # Get data
    data = get_data()
    if include_copper:
        demand, holding_costs, supplier_costs, capacity, supply_limit, Cr, Ni, Cr_required, Ni_required, CopperLimit, Cu = data
    else:
        demand, holding_costs, supplier_costs, capacity, supply_limit, Cr, Ni, Cr_required, Ni_required = data
        CopperLimit, Cu = None, None

    # Define model with or without copper
    model, x, I, z, y, Cu_removed = define_model(n_products=len(demand), n_suppliers=len(supplier_costs), n_months=len(demand[0]), include_copper=include_copper)

    # Solve the model
    set_data_and_solve(model, x, I, z, y, Cu_removed, demand, holding_costs, supplier_costs, capacity, supply_limit, Cr, Ni, Cr_required, Ni_required, CopperLimit, Cu, include_copper=include_copper)