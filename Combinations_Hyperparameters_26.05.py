#however some modifications shall be made, e.g., excel_path, etc.

import pandas as pd
import itertools

# Define the hyperparameters and their possible values
dataset_choices = ["2x", "4x"]
learning_rates = [1e-5, 1e-4, 1e-3, 1e-2]
max_epochs_options = [100, 1000]

# Generate all possible combinations
combinations = list(itertools.product(dataset_choices, learning_rates, max_epochs_options))

# Create a DataFrame to store these combinations
df = pd.DataFrame(combinations, columns=["dataset_choice", "learning_rate", "max_epochs"])

# Assign a unique job ID to each combination
df['job_ID'] = ["job_" + str(i+1) for i in range(len(df))]

import ace_tools as tools; tools.display_dataframe_to_user(name="Hyperparameter Combinations", dataframe=df)

# Save the DataFrame to an Excel file
excel_path = "/mnt/data/hyperparameter_combinations.xlsx"
df.to_excel(excel_path, index=False)

excel_path
