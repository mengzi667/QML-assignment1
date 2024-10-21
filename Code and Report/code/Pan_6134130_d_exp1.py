from gurobipy import GRB
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
from Pan_6134130_b_model import define_model, set_data_and_solve
from Pan_6134130_data import get_data_b


def run_experiment(holding_cost_combinations, capacity_variations):
    """
    Runs experiments with different holding cost combinations and capacity variations.

    Args:
        holding_cost_combinations (list of tuples): List of tuples containing different holding cost combinations.
        capacity_variations (list of int): List of different capacity values to test.

    Returns:
        list of dict: List of results for each experiment, including holding costs, capacity, total cost, and plans.
    """
    # Retrieve data for the experiment
    demand, base_holding_costs, supplier_costs, base_capacity, supply_limit, Cr, Ni, Cr_required, Ni_required = get_data_b()
    results = []

    # Iterate over each combination of holding costs and capacity variations
    for holding_costs in holding_cost_combinations:
        for capacity in capacity_variations:
            # Adjust capacity and holding costs for the current experiment
            adjusted_capacity = np.full_like(base_capacity, capacity)
            adjusted_holding_costs = np.array(holding_costs)

            # Define and solve the model with the adjusted parameters
            model, x, I, z = define_model(n_products=3, n_suppliers=5, n_months=12)
            set_data_and_solve(model, x, I, z, demand, adjusted_holding_costs, supplier_costs,
                               adjusted_capacity, supply_limit, Cr, Ni, Cr_required, Ni_required)

            # Collect results for the current experiment
            result = {
                'Holding Cost 18/10': holding_costs[0],
                'Holding Cost 18/8': holding_costs[1],
                'Holding Cost 18/0': holding_costs[2],
                'Capacity': capacity,
                'Total Cost': model.objVal if model.status == GRB.OPTIMAL else None,
                'Production Plan': np.array([[x[i, t].x for t in range(12)] for i in range(3)]) if model.status == GRB.OPTIMAL else None,
                'Inventory Plan': np.array([[I[i, t].x for t in range(12)] for i in range(3)]) if model.status == GRB.OPTIMAL else None,
                'Procurement Plan': np.array([[[z[i, j, t].x for t in range(12)] for j in range(5)] for i in range(3)]) if model.status == GRB.OPTIMAL else None
            }
            results.append(result)

    return results


def plot_results(results):
    """
    Plots the results of the experiments.

    Args:
        results (DataFrame): DataFrame containing the results of the experiments.
    """
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(3, 3, figsize=(24, 24))
    fig.suptitle('Total Cost under Different Holding Costs and Capacities', fontsize=24, y=1.02)
    cmap = sns.color_palette("YlOrRd", as_cmap=True)
    product_types = ['18/10', '18/8', '18/0']
    vmin, vmax = results['Total Cost'].min(), results['Total Cost'].max()

    # Plot heatmaps for each capacity and product type
    for i, capacity in enumerate([50, 100, 150]):
        data = results[results['Capacity'] == capacity]
        for j, product in enumerate(product_types):
            other_products = [p for p in product_types if p != product]
            pivot_data = data.pivot(index=f'Holding Cost {product}',
                                    columns=[f'Holding Cost {op}' for op in other_products],
                                    values='Total Cost')
            sns.heatmap(pivot_data, ax=axes[i, j], annot=True, fmt='.0f', cmap=cmap,
                        cbar=True, square=True, linewidths=0.5,
                        annot_kws={'fontsize': 10}, vmin=vmin, vmax=vmax)
            axes[i, j].set_title(f'Capacity: {capacity}, Product: {product}', fontsize=16, pad=20)
            axes[i, j].set_xlabel(f'Holding Cost of Other Products', fontsize=12, labelpad=10)
            axes[i, j].set_ylabel(f'Holding Cost of {product}', fontsize=12, labelpad=10)
            axes[i, j].tick_params(axis='both', which='major', labelsize=10)
            axes[i, j].set_xticklabels(axes[i, j].get_xticklabels(), rotation=45, ha='right')
            axes[i, j].collections[0].colorbar.set_label('Total Cost', fontsize=12, labelpad=10)

    plt.tight_layout()
    fig.subplots_adjust(top=0.95, wspace=0.3, hspace=0.4)
    plt.show()


def export_results_to_excel(results, filename='d_detailed_results.xlsx'):
    """
    Exports the results of the experiments to an Excel file.

    Args:
        results (list of dict): List of results for each experiment.
        filename (str): Name of the Excel file to save the results.
    """
    writer = pd.ExcelWriter(filename, engine='xlsxwriter')
    summary_df = pd.DataFrame([{k: v for k, v in r.items() if k not in ['Production Plan', 'Inventory Plan', 'Procurement Plan']} for r in results])
    summary_df.to_excel(writer, sheet_name='Summary', index=False)

    # Write detailed results for each experiment to separate sheets
    for i, result in enumerate(results):
        if result['Total Cost'] is not None:
            sheet_name = f'Experiment_{i+1}'
            workbook = writer.book
            worksheet = workbook.add_worksheet(sheet_name)
            worksheet.write('A1', 'Experiment Parameters')
            worksheet.write('A2', 'Holding Cost 18/10')
            worksheet.write('B2', result['Holding Cost 18/10'])
            worksheet.write('A3', 'Holding Cost 18/8')
            worksheet.write('B3', result['Holding Cost 18/8'])
            worksheet.write('A4', 'Holding Cost 18/0')
            worksheet.write('B4', result['Holding Cost 18/0'])
            worksheet.write('A5', 'Capacity')
            worksheet.write('B5', result['Capacity'])
            worksheet.write('A6', 'Total Cost')
            worksheet.write('B6', result['Total Cost'])

            # Write production plan
            prod_df = pd.DataFrame(result['Production Plan'], index=['18/10', '18/8', '18/0'], columns=[f'Month {i+1}' for i in range(12)])
            prod_df.to_excel(writer, sheet_name=sheet_name, startrow=8, startcol=0)

            # Write inventory plan
            inv_df = pd.DataFrame(result['Inventory Plan'], index=['18/10', '18/8', '18/0'], columns=[f'Month {i+1}' for i in range(12)])
            inv_df.to_excel(writer, sheet_name=sheet_name, startrow=14, startcol=0)

            # Write procurement plan
            proc_plan = result['Procurement Plan']
            for p in range(proc_plan.shape[0]):
                proc_df = pd.DataFrame(proc_plan[p], index=['Supplier A', 'Supplier B', 'Supplier C', 'Supplier D', 'Supplier E'], columns=[f'Month {i+1}' for i in range(12)])
                proc_df.to_excel(writer, sheet_name=sheet_name, startrow=20 + p * 8, startcol=0)

    writer.close()


def main():
    """
    Main function to run the experiments, plot the results, and export them to an Excel file.
    """
    holding_cost_variations = [5, 10, 20]
    holding_cost_combinations = list(product(holding_cost_variations, repeat=3))
    capacity_variations = [50, 100, 150]

    # Run experiments with different combinations of holding costs and capacities
    results = run_experiment(holding_cost_combinations, capacity_variations)
    results_df = pd.DataFrame([{k: v for k, v in r.items() if k not in ['Production Plan', 'Inventory Plan', 'Procurement Plan']} for r in results])

    # Print and save the summary of results
    print(results_df)
    results_df.to_excel('summary_output.xlsx', index=False)

    # Plot the results
    plot_results(results_df)

    # Export detailed results to an Excel file
    export_results_to_excel(results, 'd_detailed_results.xlsx')


if __name__ == "__main__":
    main()