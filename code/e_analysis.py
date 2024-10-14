import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read data from Excel file
file_path = 'e_result.xlsx'
df = pd.read_excel(file_path, sheet_name='Sheet1', header=None)  # Read without header

# Use the first row as the x-axis
copper_limit = df.iloc[0, 1:].tolist()

# Ensure copper_limit values are numeric and convert to percentage strings
copper_limit = [f"{float(x) * 100:.2f}%" for x in copper_limit]

# Use the remaining rows as the y-axes
electrolysis_costs = df.iloc[1, 1:].tolist()
holding_costs = df.iloc[2, 1:].tolist()
total_cost = df.iloc[3, 1:].tolist()

# Create a DataFrame
data = {
    'Copper Limit': copper_limit,
    'Electrolysis Costs': electrolysis_costs,
    'Holding Costs': holding_costs,
    'Total Cost': total_cost
}
df = pd.DataFrame(data)

# Set the style
sns.set(style="whitegrid")

# Plotting
plt.figure(figsize=(14, 10))
plt.plot(df['Copper Limit'], df['Electrolysis Costs'], label='Electrolysis Costs', marker='o', linestyle='-', color='b')
plt.plot(df['Copper Limit'], df['Holding Costs'], label='Holding Costs', marker='s', linestyle='--', color='g')
plt.plot(df['Copper Limit'], df['Total Cost'], label='Total Cost', marker='^', linestyle='-.', color='r')

plt.xlabel('Copper Limit (%)')
plt.ylabel('Cost (€)')
plt.title('Cost Analysis vs Copper Limit')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# Rotate x-axis labels
plt.xticks(rotation=45, ha='right')

# Add a tight layout to prevent clipping of tick-labels
plt.tight_layout()

plt.show()