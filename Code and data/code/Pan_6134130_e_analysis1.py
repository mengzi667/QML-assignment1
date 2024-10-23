import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_data():
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
    return demand, holding_costs, supplier_costs, capacity, supply_limit, Cr, Ni, Cr_required, Ni_required, Cu


def solve_model_with_copper_limit(copper_limit):
    data = get_data()
    demand, holding_costs, supplier_costs, capacity, supply_limit, Cr, Ni, Cr_required, Ni_required, Cu = data

    n_products = len(demand)
    n_suppliers = len(supplier_costs)
    n_months = len(demand[0])

    model = gp.Model('Copper_Analysis_Model')
    model.setParam('OutputFlag', 0)  # Suppress output

    # Variables
    x = model.addVars(n_products, n_months, vtype=GRB.CONTINUOUS, name="x")
    I = model.addVars(n_products, n_months, vtype=GRB.CONTINUOUS, name="I")
    z = model.addVars(n_products, n_suppliers, n_months, vtype=GRB.CONTINUOUS, name="z")
    y = model.addVars(n_months, vtype=GRB.BINARY, name="y")
    Cu_removed = model.addVars(n_products, n_months, vtype=GRB.CONTINUOUS, name="Cu_removed")

    # Objective
    holding_cost = gp.quicksum(holding_costs[p] * I[p, t] for p in range(n_products) for t in range(n_months))
    procurement_cost = gp.quicksum(
        supplier_costs[s] * z[p, s, t] for p in range(n_products) for s in range(n_suppliers) for t in range(n_months))
    electrolysis_cost = gp.quicksum(
        100 * y[t] + 5 * gp.quicksum(Cu_removed[p, t] for p in range(n_products)) for t in range(n_months))

    model.setObjective(holding_cost + procurement_cost + electrolysis_cost, GRB.MINIMIZE)

    # Constraints
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
            model.addConstr(gp.quicksum(Cu[s] / 100 * z[p, s, t] for s in range(n_suppliers)) - Cu_removed[
                p, t] <= copper_limit / 100 * (x[p, t] - Cu_removed[p, t]))
            model.addConstr(Cu_removed[p, t] <= y[t] * 10000)
            model.addConstr(Cu_removed[p, t] <= gp.quicksum(Cu[s] / 100 * z[p, s, t] for s in range(n_suppliers)))
            model.addConstr(gp.quicksum(z[p, s, t] for s in range(n_suppliers)) == x[p, t])
            model.addConstr(gp.quicksum(Cr[s] * z[p, s, t] for s in range(n_suppliers)) == Cr_required[p] * (
                        x[p, t] - Cu_removed[p, t]))
            model.addConstr(gp.quicksum(Ni[s] * z[p, s, t] for s in range(n_suppliers)) == Ni_required[p] * (
                        x[p, t] - Cu_removed[p, t]))

    model.optimize()

    if model.status == GRB.OPTIMAL:
        holding_cost_value = sum(holding_costs[p] * I[p, t].x for p in range(n_products) for t in range(n_months))
        electrolysis_cost_value = sum(
            100 * y[t].x + 5 * sum(Cu_removed[p, t].x for p in range(n_products)) for t in range(n_months))
        total_cost = model.objVal
        return holding_cost_value, electrolysis_cost_value, total_cost
    else:
        return None, None, None


def analyze_copper_limits():
    # Generate copper limits to test
    copper_limits = np.arange(0, 3.1, 0.1)  # From 0% to 3% in 0.1% steps

    # Store results
    results = []

    for limit in copper_limits:
        holding_cost, electrolysis_cost, total_cost = solve_model_with_copper_limit(limit)
        if holding_cost is not None:
            results.append({
                'copper_limit': limit,
                'holding_cost': holding_cost,
                'electrolysis_cost': electrolysis_cost,
                'total_cost': total_cost
            })

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Create visualization
    plt.figure(figsize=(15, 8))
    plt.plot(df['copper_limit'], df['electrolysis_cost'], 'b.-', label='Electrolysis Costs')
    plt.plot(df['copper_limit'], df['holding_cost'], 'g--', label='Holding Costs')
    plt.plot(df['copper_limit'], df['total_cost'], 'r-.', label='Total Cost')

    plt.xlabel('Copper Limit (%)')
    plt.ylabel('Cost (€)')
    plt.title('Cost Analysis vs Copper Limit')
    plt.grid(True)
    plt.legend()

    # Save results to CSV
    df.to_csv('copper_limit_analysis.csv', index=False)
    plt.savefig('copper_limit_analysis.png')

    return df


if __name__ == "__main__":
    results_df = analyze_copper_limits()
    print("\nAnalysis Results:")
    print("\nCost breakdown at selected copper limits:")
    for limit in [0.0, 1.0, 2.0, 3.0]:
        row = results_df[results_df['copper_limit'] == limit].iloc[0]
        print(f"\nCopper limit: {limit}%")
        print(f"Holding costs: {row['holding_cost']:.2f} €")
        print(f"Electrolysis costs: {row['electrolysis_cost']:.2f} €")
        print(f"Total costs: {row['total_cost']:.2f} €")