import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import math
import io
from fpdf import FPDF

# --- Page config ---
st.set_page_config(page_title="AI Project Cost Estimator", layout="wide")
st.write("App Version: 2025-05-19-v15")

# --- Load model ---
model = joblib.load("d:/Report/models/best_xgb_pipeline.pkl")

# --- Roles by project type ---
roles_by_type = {
    "IT": ["Developers", "Designers", "QA_Engineers"],
    "Construction": ["Labourers", "Supervisors"],
    "Event": ["Coordinators", "Staff", "Technicians"],
    "Marketing": ["Creatives", "Analysts"],
    "Research": ["Scientists", "Assistants"]
}

st.title("💡 AI-Powered Project Cost Estimation")

# --- Project Details Input ---
with st.expander("Project Details", expanded=True):
    col1, col2, col3 = st.columns([2,2,2])
    with col1:
        project_type = st.selectbox("Project Type", list(roles_by_type.keys()),
                                    help="Select the main category of your project.")
        duration_days = st.number_input("Duration (Days)", min_value=1, value=10,
                                        help="Enter the total number of days for the project.")
    with col2:
        hourly_rate = st.number_input("Hourly Rate (₹/hr)", min_value=0, value=500,
                                      help="Average hourly cost per staff.")
    with col3:
        misc_cost = st.number_input("Miscellaneous Cost (₹)", min_value=0, value=0,
                                    help="Other non-labor costs.")
    st.number_input("Material Cost (₹)", min_value=0, value=0,
                    help="Cost of materials used.", key="material_cost")

material_cost = st.session_state.material_cost

# --- Staffing Inputs ---
st.subheader("👥 Staffing / Role Counts")
roles = roles_by_type[project_type]
cols = st.columns(len(roles))
role_counts = {}
for i, role in enumerate(roles):
    label = role.replace("_", " ")
    role_counts[role] = cols[i].number_input(f"{label}", min_value=0, value=0,
                                            help=f"Number of {label}s assigned.")

# --- Prepare model input ---
input_dict = {
    "Project_Type": project_type,
    "Duration_Days": duration_days,
    "Hourly_Rate": hourly_rate,
    "Misc_Cost": misc_cost,
    "Material_Cost": material_cost
}
all_roles = sorted({r for roles in roles_by_type.values() for r in roles})
for role in all_roles:
    input_dict[role] = role_counts.get(role, 0)

input_df = pd.DataFrame([input_dict])

# --- Risk assessment function ---
def assess_risk(duration, staff_total, misc, material, predicted, hourly_rate):
    risks = []
    labor_cost = duration * 8 * hourly_rate * staff_total if staff_total > 0 else 0
    labor_pct = labor_cost / predicted * 100 if predicted else 0
    misc_pct = misc / predicted * 100 if predicted else 0
    mat_pct  = material / predicted * 100 if predicted else 0

    if staff_total == 0 and predicted > 0:
        risks.append("No staff but non-zero cost")
        level, icon = "High", "❌"
    elif labor_pct > 60:
        risks.append("Labor > 60% of total cost")
        level, icon = "High", "❌"
    elif duration < 7 and staff_total > 5:
        risks.append("Short timeline & many staff")
        level, icon = "High", "❌"
    elif duration > 30 and staff_total < 2:
        risks.append("Long project duration with low staff")
        level, icon = "Medium", "⚠️"
    elif misc_pct > 50 or mat_pct > 50:
        risks.append("Non-labor costs dominate")
        level, icon = "Medium", "⚠️"
    else:
        risks.append("No major risks detected")
        level, icon = "Low", "✅"

    return f"{icon} {level}", risks

def generate_pdf_report(input_data, predicted_cost, risk_level, risk_reasons):
    # 1. Define only the roles you want per project type
    relevant_roles = {
        "IT":        ["Developers", "Designers", "QA_Engineers"],
        "Construction": ["Labourers", "Supervisors"],
        "Event":     ["Coordinators", "Staff", "Technicians"],
        "Marketing": ["Creatives", "Analysts"],
        "Research":  ["Scientists", "Assistants"]
    }

    # 2. Grab the list for this run
    project_type = input_data["Project_Type"]
    roles_to_include = relevant_roles.get(project_type, [])

    # 3. Prepare the PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Title
    pdf.cell(0, 10, "Project Cost Estimation Report", ln=True, align='C')
    pdf.ln(10)

    # Project details
    pdf.cell(0, 10, "Project Details:", ln=True)
    for key in ["Project_Type", "Duration_Days", "Hourly_Rate", "Misc_Cost", "Material_Cost"]:
        pdf.cell(0, 8, f"{key}: {input_data[key]}", ln=True)

    pdf.ln(4)
    # Staffing — only show the roles relevant to project_type
    pdf.cell(0, 10, "Staffing (Relevant Roles):", ln=True)
    for role in roles_to_include:
        pdf.cell(0, 8, f"{role}: {input_data.get(role, 0)}", ln=True)

    pdf.ln(10)
    # Cost and risk
    pdf.cell(0, 10, f"AI Estimated Cost: Rs. {predicted_cost:,.2f}", ln=True)
    pdf.ln(6)
    # Strip emojis from risk_level
    clean_risk = ''.join(ch for ch in risk_level if ord(ch) < 128)
    pdf.cell(0, 10, f"Risk Analysis: {clean_risk}", ln=True)
    for reason in risk_reasons:
        pdf.cell(0, 8, f"- {reason}", ln=True)

    # Return BytesIO
    buf = io.BytesIO()
    pdf.output(buf)
    buf.seek(0)
    return buf

# --- Excel report generator ---
def generate_excel_report(input_data, predicted_cost, risk_level, risk_reasons):
    output = io.BytesIO()

    # Define project-specific roles
    relevant_roles = {
        "IT": ["Developers", "Designers", "QA_Engineers"],
        "Construction": ["Labourers", "Supervisors"],
        "Event": ["Coordinators", "Staff", "Technicians"],
        "Marketing": ["Creatives", "Analysts"],
        "Research": ["Scientists", "Assistants"]
    }

    project_type = input_data.get("Project_Type", "")
    roles_to_include = relevant_roles.get(project_type, [])

    # General fields always shown
    base_fields = ["Project_Type", "Duration_Days", "Hourly_Rate", "Misc_Cost", "Material_Cost"]

    # Filter to base + project-relevant roles only
    filtered_data = {k: input_data[k] for k in base_fields + roles_to_include}

    # Add prediction and risk info to the same data
    filtered_data["Predicted Cost"] = f"₹{predicted_cost:,.2f}"
    filtered_data["Risk Level"] = risk_level
    filtered_data["Risk Reasons"] = ", ".join(risk_reasons)

    # Convert to DataFrame
    df_single = pd.DataFrame(list(filtered_data.items()), columns=['Feature', 'Value'])

    # Save to Excel in memory (single sheet)
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_single.to_excel(writer, sheet_name='Project Report', index=False)

    output.seek(0)
    return output


    # Save to Excel in memory
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_input.to_excel(writer, sheet_name='Project Details', index=False)
        df_result.to_excel(writer, sheet_name='Prediction Result', index=False)

    output.seek(0)
    return output

# --- Main Estimate button and logic ---
total_staff = sum(role_counts.values())
can_estimate = total_staff > 0 and duration_days > 0 and hourly_rate >= 0

if not can_estimate:
    st.info("Please enter valid staffing, duration and hourly rate to enable estimation.")

estimate_btn = st.button("🧮 Estimate Project Cost", disabled=not can_estimate)

if estimate_btn:
    with st.spinner('Estimating cost...'):
        # Prediction & base calculation
        base_cost = total_staff * duration_days * 8 * hourly_rate + misc_cost + material_cost
        predicted_cost = model.predict(input_df)[0]

        st.success(f"💰 AI Estimated Cost: ₹{predicted_cost:,.2f}")

        # Risk analysis
        risk_level, reasons = assess_risk(duration_days, total_staff, misc_cost, material_cost, predicted_cost, hourly_rate)
        st.markdown(f"### 🔎 Risk Analysis: {risk_level}")
        for r in reasons:
            st.markdown(f"- {r}")

        # Cost breakdown visualization
        st.subheader("📊 Cost Breakdown")
        labels = ["Labor Cost", "Miscellaneous Cost", "Material Cost"]
        labor_cost = duration_days * 8 * hourly_rate * total_staff
        costs = [labor_cost, misc_cost, material_cost]
        costs = [0 if (x is None or (isinstance(x, float) and math.isnan(x)) or x < 0) else x for x in costs]
        total_cost = sum(costs)

        col1, col2 = st.columns(2)

        with col1:
            if total_cost == 0:
                st.warning("All cost components are zero or invalid — cannot show pie chart.")
            else:
                fig1, ax1 = plt.subplots()
                ax1.pie(costs, labels=labels, autopct='%1.1f%%', startangle=140, colors=['#66b3ff','#99ff99','#ffcc99'])
                ax1.axis('equal')
                plt.title("Cost Component Distribution")
                st.pyplot(fig1)

        with col2:
            st.subheader("Staffing Breakdown")
            staff_roles = {role.replace("_", " "): count for role, count in role_counts.items() if count > 0}
            if staff_roles:
                staff_df = pd.DataFrame.from_dict(staff_roles, orient='index', columns=['Count'])
                st.bar_chart(staff_df)
            else:
                st.write("No staff assigned to the project.")

        # Estimated cost vs duration chart
        st.subheader("📈 Estimated Cost vs Duration (Simulated)")
        durations = np.arange(1, 61)
        estimated_costs = []
        for d in durations:
            lc = d * 8 * hourly_rate * total_staff
            total = lc + misc_cost + material_cost
            estimated_costs.append(total)

        cost_df = pd.DataFrame({"Duration (days)": durations, "Estimated Cost (₹)": estimated_costs}).set_index("Duration (days)")
        st.line_chart(cost_df)

        # --- Download buttons ---
        pdf_buffer = generate_pdf_report(input_dict, predicted_cost, risk_level, reasons)
        excel_buffer = generate_excel_report(input_dict, predicted_cost, risk_level, reasons)

        st.download_button(
            label="📄 Download Report as PDF",
            data=pdf_buffer,
            file_name="project_cost_estimate_report.pdf",
            mime="application/pdf"
        )

        st.download_button(
            label="📊 Download Report as Excel",
            data=excel_buffer,
            file_name="project_cost_estimate_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
