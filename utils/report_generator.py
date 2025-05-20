import pandas as pd
import numpy as np

np.random.seed(42)

# Load existing dataset
df_existing = pd.read_csv('d:/Report/data/project_cost_data.csv')

# Define roles by project type
project_types = {
    "IT": ["Developers", "Designers", "QA_Engineers"],
    "Construction": ["Labourers", "Supervisors"],
    "Event": ["Coordinators", "Staff", "Technicians"],
    "Marketing": ["Creatives", "Analysts"],
    "Research": ["Scientists", "Assistants"]
}

all_roles = sorted({r for roles in project_types.values() for r in roles})
new_data = []

# Generate 500 new samples with low staff and short durations
for _ in range(500):
    project_type = np.random.choice(list(project_types.keys()))
    duration_days = np.random.randint(1, 16)  # 1–15 days
    hourly_rate = np.random.randint(300, 600)  # ₹300–₹600/hr
    misc_cost = np.random.randint(0, 5000)  # Low misc costs
    material_cost = np.random.randint(0, 10000)  # Low material costs

    role_counts = {role: 0 for role in all_roles}
    num_active_roles = np.random.randint(1, 3)  # 1–2 roles
    active_roles = np.random.choice(project_types[project_type], num_active_roles, replace=False)
    for role in active_roles:
        role_counts[role] = np.random.randint(1, 3)  # 1–2 staff per role

    total_staff = sum(role_counts.values())
    total_hours = duration_days * 8
    labor_cost = total_staff * total_hours * hourly_rate
    total_cost = labor_cost + misc_cost + material_cost + np.random.normal(0, 5000)

    sample = {
        "Project_Type": project_type,
        "Duration_Days": duration_days,
        "Hourly_Rate": hourly_rate,
        "Misc_Cost": misc_cost,
        "Material_Cost": material_cost,
        **role_counts,
        "Total_Cost": max(total_cost, 0)
    }
    new_data.append(sample)

# Create new DataFrame
df_new = pd.DataFrame(new_data)

# Combine with existing dataset
df_combined = pd.concat([df_existing, df_new], ignore_index=True)

# Save augmented dataset
df_combined.to_csv('d:/Report/data/project_cost_data_augmented.csv', index=False)
print("Augmented dataset saved to 'd:/Report/data/project_cost_data_augmented.csv'")
print(f"Original rows: {len(df_existing)}, New rows: {len(df_new)}, Total: {len(df_combined)}")