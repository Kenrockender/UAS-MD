"""
app.py
======
Streamlit web deployment for the Credit Score classifier.

Run locally:
    streamlit run app.py

Deploy (public URL) on Streamlit Community Cloud:
    push this folder to GitHub, then point share.streamlit.io at app.py.
"""

import os
import pandas as pd
import streamlit as st

from inference import CreditScorePredictor, TEST_CASES

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(page_title="Credit Score Predictor", page_icon="💳", layout="wide")

OCCUPATIONS = [
    "Accountant", "Architect", "Developer", "Doctor", "Engineer", "Entrepreneur",
    "Journalist", "Lawyer", "Manager", "Mechanic", "Media_Manager", "Musician",
    "Scientist", "Teacher", "Writer",
]
CREDIT_MIX = ["Good", "Standard", "Bad"]
MIN_AMOUNT = ["No", "Yes", "NM"]
PAY_BEHAVIOUR = [
    "High_spent_Large_value_payments", "High_spent_Medium_value_payments",
    "High_spent_Small_value_payments", "Low_spent_Large_value_payments",
    "Low_spent_Medium_value_payments", "Low_spent_Small_value_payments",
]
SCORE_COLOR = {"Good": "🟢", "Standard": "🟡", "Poor": "🔴"}


@st.cache_resource
def get_predictor():
    return CreditScorePredictor(os.path.join(BASE_DIR, "models", "credit_score_model.pkl"))


predictor = get_predictor()

st.title("💳 Credit Score Classification")
st.caption(
    f"Best model: **{predictor.model_name}** · "
    f"test accuracy **{predictor.metrics.get('accuracy', 0):.2%}** · "
    f"F1 (weighted) **{predictor.metrics.get('f1_weighted', 0):.2%}**"
)

# Allow preloading a test case via URL, e.g. ?case=Good (handy for screenshots)
_case_param = st.query_params.get("case")
if _case_param:
    _key = f"Expected: {_case_param}"
    if _key in TEST_CASES and st.session_state.get("_loaded_param") != _case_param:
        st.session_state["preset"] = TEST_CASES[_key]
        st.session_state["_loaded_param"] = _case_param

# Quick-load a representative test case (handy for grading screenshots)
preset = st.session_state.get("preset", {})
with st.sidebar:
    st.header("Customer Inputs")
    st.markdown("**Quick test cases**")
    cols = st.columns(3)
    for col, name in zip(cols, TEST_CASES):
        label = name.replace("Expected: ", "")
        if col.button(label, width="stretch"):
            st.session_state["preset"] = TEST_CASES[name]
            st.rerun()
    preset = st.session_state.get("preset", {})

    def g(key, default):
        return preset.get(key, default)

    st.subheader("Demographics & Income")
    age = st.number_input("Age", 0, 100, int(g("Age", 33)))
    occupation = st.selectbox("Occupation", OCCUPATIONS,
                              index=OCCUPATIONS.index(g("Occupation", "Engineer"))
                              if g("Occupation", "Engineer") in OCCUPATIONS else 0)
    annual_income = st.number_input("Annual Income", 0.0, 300000.0, float(g("Annual_Income", 37000.0)))
    monthly_salary = st.number_input("Monthly Inhand Salary", 0.0, 50000.0, float(g("Monthly_Inhand_Salary", 3100.0)))

    st.subheader("Accounts & Loans")
    num_bank = st.number_input("Num Bank Accounts", 0, 20, int(g("Num_Bank_Accounts", 5)))
    num_card = st.number_input("Num Credit Cards", 0, 20, int(g("Num_Credit_Card", 5)))
    interest = st.number_input("Interest Rate", 0, 50, int(g("Interest_Rate", 13)))
    num_loan = st.number_input("Num of Loans", 0, 20, int(g("Num_of_Loan", 3)))
    total_emi = st.number_input("Total EMI per month", 0.0, 100000.0, float(g("Total_EMI_per_month", 70.0)))

    st.subheader("Payment Behaviour")
    delay = st.number_input("Delay from due date (days)", 0, 100, int(g("Delay_from_due_date", 18)))
    num_delayed = st.number_input("Num of Delayed Payments", 0, 100, int(g("Num_of_Delayed_Payment", 14)))
    changed_limit = st.number_input("Changed Credit Limit", -50.0, 50.0, float(g("Changed_Credit_Limit", 9.4)))
    num_inquiry = st.number_input("Num Credit Inquiries", 0, 50, int(g("Num_Credit_Inquiries", 6)))
    min_amount = st.selectbox("Payment of Min Amount", MIN_AMOUNT,
                              index=MIN_AMOUNT.index(g("Payment_of_Min_Amount", "No")))
    pay_behaviour = st.selectbox("Payment Behaviour", PAY_BEHAVIOUR,
                                 index=PAY_BEHAVIOUR.index(g("Payment_Behaviour", PAY_BEHAVIOUR[4])))

    st.subheader("Credit Profile")
    credit_mix = st.selectbox("Credit Mix", CREDIT_MIX,
                              index=CREDIT_MIX.index(g("Credit_Mix", "Standard")))
    outstanding = st.number_input("Outstanding Debt", 0.0, 10000.0, float(g("Outstanding_Debt", 1170.0)))
    utilization = st.number_input("Credit Utilization Ratio", 0.0, 100.0, float(g("Credit_Utilization_Ratio", 32.3)))
    history_age = st.text_input("Credit History Age", g("Credit_History_Age", "18 Years and 5 Months"))
    invested = st.number_input("Amount Invested Monthly", 0.0, 10000.0, float(g("Amount_invested_monthly", 136.0)))
    balance = st.number_input("Monthly Balance", 0.0, 100000.0, float(g("Monthly_Balance", 336.0)))

record = {
    "Age": age, "Occupation": occupation, "Annual_Income": annual_income,
    "Monthly_Inhand_Salary": monthly_salary, "Num_Bank_Accounts": num_bank,
    "Num_Credit_Card": num_card, "Interest_Rate": interest, "Num_of_Loan": num_loan,
    "Delay_from_due_date": delay, "Num_of_Delayed_Payment": num_delayed,
    "Changed_Credit_Limit": changed_limit, "Num_Credit_Inquiries": num_inquiry,
    "Credit_Mix": credit_mix, "Outstanding_Debt": outstanding,
    "Credit_Utilization_Ratio": utilization, "Credit_History_Age": history_age,
    "Payment_of_Min_Amount": min_amount, "Total_EMI_per_month": total_emi,
    "Amount_invested_monthly": invested, "Payment_Behaviour": pay_behaviour,
    "Monthly_Balance": balance,
}

result = predictor.predict(record)
score = result["credit_score"]

c1, c2 = st.columns([1, 1])
with c1:
    st.subheader("Prediction")
    st.metric("Predicted Credit Score", f"{SCORE_COLOR.get(score,'')} {score}")
    st.metric("Confidence", f"{result['confidence']:.2%}")
    if score == "Good":
        st.success("Low credit risk — strong repayment profile.")
    elif score == "Standard":
        st.warning("Moderate credit risk — average repayment profile.")
    else:
        st.error("High credit risk — weak repayment profile.")

with c2:
    st.subheader("Class Probabilities")
    prob_df = pd.DataFrame(
        {"Probability": result["probabilities"]}
    ).reindex(["Poor", "Standard", "Good"])
    st.bar_chart(prob_df)

st.divider()
st.subheader("Input Summary")
st.dataframe(pd.DataFrame([record]), width="stretch")
