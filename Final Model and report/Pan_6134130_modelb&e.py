import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd


# data part
def get_data_b():
    """
    Generates and returns data related to monthly product demands, costs, capacities, and material requirements.
    """
    n_months, n_products, n_suppliers = 12, 3, 5

    # Monthly demand for each product
    demand = np.array([
        [25, 25, 0, 0, 0, 50, 12, 0, 10, 10, 45, 99],
        [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
        [5, 20, 80, 25, 50, 125, 150, 80, 40, 35, 3, 100]
    ])

    # Holding costs for each product
    holding_costs = np.array([20, 10, 5])
    # Supplier costs for each supplier
    supplier_costs = np.array([5, 10, 9, 7, 8.5])
    # Monthly production capacity
    capacity = np.full(n_months, 100)
    # Supply limit for each supplier
    supply_limit = np.array([90, 30, 50, 70, 20])
    # Chromium content for each supplier
    Cr = np.array([18, 25, 15, 14, 0])
    # Nickel content for each supplier
    Ni = np.array([0, 15, 10, 16, 10])
    # Required Chromium content for each product
    Cr_required = np.array([18, 18, 18])
    # Required Nickel content for each product
    Ni_required = np.array([10, 8, 0])

    return demand, holding_costs, supplier_costs, capacity, supply_limit, Cr, Ni, Cr_required, Ni_required

def get_data_e():
    """
    Generates and returns data related to monthly product demands, costs, capacities, and material requirements, including copper.
    """
    n_months, n_products, n_suppliers = 12, 3, 5

    # Monthly demand for each product
    demand = np.array([
        [25, 25, 0, 0, 0, 50, 12, 0, 10, 10, 45, 99],
        [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
        [5, 20, 80, 25, 50, 125, 150, 80, 40, 35, 3, 100]
    ])

    # Holding costs for each product
    holding_costs = np.array([20, 10, 5])
    # Supplier costs for each supplier
    supplier_costs = np.array([5, 10, 9, 7, 8.5])
    # Monthly production capacity
    capacity = np.full(n_months, 100)
    # Supply limit for each supplier
    supply_limit = np.array([90, 30, 50, 70, 20])
    # Chromium content for each supplier
    Cr = np.array([18, 25, 15, 14, 0])
    # Nickel content for each supplier
    Ni = np.array([0, 15, 10, 16, 10])
    # Required Chromium content for each product
    Cr_required = np.array([18, 18, 18])
    # Required Nickel content for each product
    Ni_required = np.array([10, 8, 0])
    # Copper limit for each month
    CopperLimit = np.full(n_months, 50)
    # Copper content for each supplier
    Cu = np.array([5, 10, 15, 20, 25])

    return demand, holding_costs, supplier_costs, capacity, supply_limit, Cr, Ni, Cr_required, Ni_required, CopperLimit, Cu


# model part
def define_model(n_products, n_suppliers, n_months, tolerance=1e-6):
    """
    Defines the basic model without copper constraints.
    """
    model = gp.Model('Three_Subscript_Model')
    model.setParam('OptimalityTol', tolerance)
    model.setParam('FeasibilityTol', tolerance)

    # Decision variables
    x = model.addVars(n_products, n_months, vtype=GRB.CONTINUOUS, name="x")
    I = model.addVars(n_products, n_months, vtype=GRB.CONTINUOUS, name="I")
    z = model.addVars(n_products, n_suppliers, n_months, vtype=GRB.CONTINUOUS, name="z")

    return model, x, I, z

def define_model_with_copper(n_products, n_suppliers, n_months, tolerance=1e-6):
    """
    Defines the model with copper constraints.
    """
    model = gp.Model('Three_Subscript_Model_With_Copper')
    model.setParam('OptimalityTol', tolerance)
    model.setParam('FeasibilityTol', tolerance)

    # Decision variables
    x = model.addVars(n_products, n_months, vtype=GRB.CONTINUOUS, name="x")
    I = model.addVars(n_products, n_months, vtype=GRB.CONTINUOUS, name="I")
    z = model.addVars(n_products, n_suppliers, n_months, vtype=GRB.CONTINUOUS, name="z")
    y = model.addVars(n_months, vtype=GRB.BINARY, name="y")
    Cu_removed = model.addVars(n_products, n_suppliers, n_months, vtype=GRB.CONTINUOUS, name="Cu_removed")

    return model, x, I, z, y, Cu_removed

def set_data_and_solve(model, x, I, z, demand, holding_costs, supplier_costs, capacity, supply_limit, Cr, Ni, Cr_required, Ni_required):
    """
    Sets the data and solves the basic model without copper constraints.
    """
    n_products = len(demand)
    n_suppliers = len(supplier_costs)
    n_months = len(demand[0])

    # Objective function: minimize holding and supplier costs
    objective = gp.quicksum(holding_costs[p] * I[p, t] for p in range(n_products) for t in range(n_months)) + \
                gp.quicksum(supplier_costs[s] * z[p, s, t] for p in range(n_products) for s in range(n_suppliers) for t in range(n_months))
    model.setObjective(objective, GRB.MINIMIZE)

    # Constraints
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

    # Optimize the model
    model.optimize()

    # Check if the model found an optimal solution
    if model.status == GRB.OPTIMAL:
        print(f"Optimal objective value: {model.objVal} EURO")

        # Extract the solution
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

        # Create dataframes for better visualization
        products = [f'Product {p + 1}' for p in range(n_products)]
        suppliers = [f'Supplier {s + 1}' for s in range(n_suppliers)]
        months = [f'Month {t + 1}' for t in range(n_months)]

        production_df = pd.DataFrame(production_plan, index=products, columns=months)
        inventory_df = pd.DataFrame(inventory_plan, index=products, columns=months)

        purchase_df_dict = {}
        for p in range(n_products):
            purchase_df_dict[products[p]] = pd.DataFrame(purchase_plan[p], index=suppliers, columns=months)

        # Print the results
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
    """
    Sets the data and solves the model with copper constraints.
    """
    n_products = len(demand)
    n_suppliers = len(supplier_costs)
    n_months = len(demand[0])

    # Objective function: minimize holding, supplier, and electrolysis costs
    objective = gp.quicksum(holding_costs[p] * I[p, t] for p in range(n_products) for t in range(n_months)) + \
                gp.quicksum(supplier_costs[s] * z[p, s, t] for p in range(n_products) for s in range(n_suppliers) for t in range(n_months)) + \
                gp.quicksum(100 * y[t] + 5 * gp.quicksum(Cu_removed[p, s, t] for p in range(n_products) for s in range(n_suppliers)) for t in range(n_months))

    model.setObjective(objective, GRB.MINIMIZE)

    # Constraints
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
            model.addConstr(gp.quicksum(Cu[s] * z[p, s, t] for s in range(n_suppliers)) - gp.quicksum(Cu_removed[p, s, t] for s in range(n_suppliers)) <= CopperLimit[t] * (x[p, t]-gp.quicksum(Cu_removed[p, s, t] for s in range(n_suppliers))))
            for s in range(n_suppliers):
                model.addConstr(Cu_removed[p, s, t] <= Cu[s] * z[p, s, t])
                model.addConstr(Cu_removed[p, s, t] <= y[t] * Cu[s] * z[p, s, t])

    # Optimize the model
    model.optimize()

    # Check if the model found an optimal solution
    if model.status == GRB.OPTIMAL:
        print(f"Optimal objective value: {model.objVal} EURO")

        # Extract the solution
        production_plan = np.zeros((n_products, n_months))
        inventory_plan = np.zeros((n_products, n_months))
        purchase_plan = np.zeros((n_products, n_suppliers, n_months))
        electrolysis_plan = np.zeros(n_months)
        copper_removed_plan = np.zeros((n_products, n_suppliers, n_months))
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
                    copper_removed_plan[p, s, t] = Cu_removed[p, s, t].x

        for t in range(n_months):
            electrolysis_plan[t] = y[t].x
            electrolysis_costs_total += 100 * y[t].x

        # Create dataframes for better visualization
        products = [f'Product {p + 1}' for p in range(n_products)]
        suppliers = [f'Supplier {s + 1}' for s in range(n_suppliers)]
        months = [f'Month {t + 1}' for t in range(n_months)]

        production_df = pd.DataFrame(production_plan, index=products, columns=months)
        inventory_df = pd.DataFrame(inventory_plan, index=products, columns=months)

        purchase_df_dict = {}
        for p in range(n_products):
            purchase_df_dict[products[p]] = pd.DataFrame(purchase_plan[p], index=suppliers, columns=months)

        copper_removed_df = pd.DataFrame(copper_removed_plan.reshape(n_products * n_suppliers, n_months),
                                         index=pd.MultiIndex.from_product([products, suppliers]),
                                         columns=months)
        electrolysis_df = pd.DataFrame({
            'Electrolysis Used': electrolysis_plan
        }, index=months)

        # Print the results
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
    # Choose the model to run based on user input
    model_choice = input("Enter 'b' to run the basic model or 'e' to run the model with copper: ").strip().lower()

    if model_choice == 'b':
        # Get data for the basic model
        demand, holding_costs, supplier_costs, capacity, supply_limit, Cr, Ni, Cr_required, Ni_required = get_data_b()
        # Define and solve the basic model
        model, x, I, z = define_model(n_products=len(demand), n_suppliers=len(supplier_costs), n_months=len(demand[0]))
        set_data_and_solve(model, x, I, z, demand, holding_costs, supplier_costs, capacity, supply_limit, Cr, Ni, Cr_required, Ni_required)
    elif model_choice == 'e':
        # Get data for the model with copper constraints
        demand, holding_costs, supplier_costs, capacity, supply_limit, Cr, Ni, Cr_required, Ni_required, CopperLimit, Cu = get_data_e()
        # Define and solve the model with copper constraints
        model, x, I, z, y, Cu_removed = define_model_with_copper(n_products=len(demand), n_suppliers=len(supplier_costs), n_months=len(demand[0]))
        set_data_and_solve_with_copper(model, x, I, z, y, Cu_removed, demand, holding_costs, supplier_costs, capacity, supply_limit, Cr, Ni, Cr_required, Ni_required, CopperLimit, Cu)
    else:
        print("Invalid input. Please enter 'b' or 'e'.")