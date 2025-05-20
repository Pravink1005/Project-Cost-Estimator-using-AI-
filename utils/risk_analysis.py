def calculate_risk_score(df_row):
    # Rule-based sample logic
    score = 0

    if df_row["Duration"].values[0] > 180:
        score += 30

    if df_row["Material_Cost"].values[0] > 500000:
        score += 30

    if df_row["Developers"].values[0] > 10:
        score += 20

    if df_row["Project_Type"].values[0] == "Construction":
        score += 10  # More prone to inflation

    # Cap at 100
    score = min(score, 100)

    # Risk level
    if score < 40:
        label = "Low"
    elif score < 70:
        label = "Medium"
    else:
        label = "High"

    return score, label
