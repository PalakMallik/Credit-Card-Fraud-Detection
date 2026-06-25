import streamlit as st
import numpy as np
import joblib
import datetime

st.set_page_config(page_title='Credit Card Fraud Detector', page_icon='💳', layout='centered')
st.title('💳 Credit Card Fraud Detection')

@st.cache_resource
def load_artifacts():
    model         = joblib.load('CreditCard_Fraud_Advanced_Model.pkl')
    scaler        = joblib.load('scaler.pkl')
    le_cat        = joblib.load('le_category.pkl')
    le_gen        = joblib.load('le_gender.pkl')
    cat_avg       = joblib.load('cat_avg_amt.pkl')
    ratio_cap     = joblib.load('amt_ratio_cap.pkl')
    threshold     = joblib.load('fraud_threshold.pkl')
    return model, scaler, le_cat, le_gen, cat_avg, ratio_cap, threshold

model, scaler, le_cat, le_gen, cat_avg, ratio_cap, THRESHOLD = load_artifacts()

def rule_based_score(amt, cat_avg_val, is_night):
    """Hard rules for extreme amounts that ML might miss"""
    ratio = amt / cat_avg_val
    if   ratio > 20:  score = 0.95
    elif ratio > 10:  score = 0.80
    elif ratio > 5:   score = 0.65
    elif ratio > 3:   score = 0.40
    elif ratio > 1.5: score = 0.20
    else:             score = 0.05
    if is_night:
        score = min(score * 1.3, 0.99)
    return score

# ── UI ──────────────────────────────────────────
st.subheader('Transaction Details')

amt           = st.number_input('Transaction Amount ($)', min_value=0.01,
                                 max_value=100000.0, value=150.0, step=1.0)
category_name = st.selectbox('Transaction Category', sorted(le_cat.classes_))
gender_disp   = st.selectbox('Gender', ['Female', 'Male'])
gender_raw    = 'M' if gender_disp == 'Male' else 'F'

col1, col2 = st.columns(2)
with col1:
    trans_date = st.date_input('Transaction Date', datetime.date.today(), format='DD/MM/YYYY')
with col2:
    trans_time = st.time_input('Transaction Time', datetime.time(12, 0))

# Live ratio hint
avg_val   = cat_avg.get(category_name, 100.0)
ratio_now = amt / avg_val
if   ratio_now <= 1.5: icon = '🟢'
elif ratio_now <= 5:   icon = '🟡'
elif ratio_now <= 10:  icon = '🟠'
else:                  icon = '🔴'
st.caption(f'{icon} Avg {category_name}: ${avg_val:.0f} | Your amount is {ratio_now:.1f}x the average')

# ── Prediction ──────────────────────────────────
if st.button('Predict Fraud 🔍'):
    trans_hour  = trans_time.hour
    trans_month = trans_date.month
    is_night    = 1 if (trans_hour >= 23 or trans_hour <= 5) else 0

    # ML prediction
    cat_enc          = le_cat.transform([category_name])[0]
    gen_enc          = le_gen.transform([gender_raw])[0]
    amt_log          = np.log1p(amt)
    amt_ratio_capped = min(ratio_now, ratio_cap)
    amt_ratio_log    = np.log1p(amt_ratio_capped)

    inp        = np.array([[cat_enc, amt_log, amt_ratio_log, gen_enc, trans_hour, trans_month, is_night]])
    ml_score   = model.predict_proba(scaler.transform(inp))[0][1]

    # Rule-based score
    rule_score = rule_based_score(amt, avg_val, is_night)

    # Final = worst case of both
    final_score = max(ml_score, rule_score)
    risk_pct    = final_score * 100

    st.divider()
    st.progress(min(int(risk_pct), 100), text=f'Risk Score: {risk_pct:.1f}%')

    if final_score >= 0.5:
        st.error(f'🚨 FRAUD Transaction Detected! — Risk: {risk_pct:.1f}%')
        reasons = []
        if ratio_now > 5:
            reasons.append(f'Amount ${amt:.0f} is **{ratio_now:.1f}x** the normal average (${avg_val:.0f})')
        if is_night:
            reasons.append('Transaction at night (11pm–5am) — high-risk window')
        if ml_score > 0.4:
            reasons.append(f'ML model also flagged this (score: {ml_score*100:.0f}%)')
        for r in reasons:
            st.write(f'• {r}')

    elif final_score >= 0.3:
        st.warning(f'⚠️ Suspicious Transaction — Risk: {risk_pct:.1f}%')
        st.write(f'Amount is {ratio_now:.1f}x the category average. Verification recommended.')
    else:
        st.success(f'✅ GENUINE Transaction — Risk: {risk_pct:.1f}%')
        st.write(f'Amount is within normal range for {category_name}.')
