import pandas as pd
import numpy as np
import streamlit as st
import mlflow
import mlflow.pyfunc

st.set_page_config(page_title="Hospital Readmission (30d)", page_icon="🏥", layout="wide")
st.title("🏥 Hospital Readmission (30-day)")

with st.sidebar:
    st.header("Model")
    model_uri = st.text_input("MLflow model URI", "models:/hospital_readmission/Production")
    if st.button("Load model"):
        try:
            st.session_state.model = mlflow.pyfunc.load_model(model_uri)
            st.success("Model loaded.")
        except Exception as e:
            st.error(f"Load failed: {e}")

if "model" not in st.session_state:
    st.session_state.model = None

tab1, tab2 = st.tabs(["Single Entry", "Batch CSV"])

with tab1:
    st.subheader("Single patient features")
    # minimal but schema-true fields (match training columns after enrich_and_clean)
    age_bucket = st.selectbox("age (bucket)", ["[0-10)","[10-20)","[20-30)","[30-40)","[40-50)","[50-60)","[60-70)","[70-80)","[80-90)","[90-100)"], index=6)
    gender = st.selectbox("gender", ["Female","Male","Unknown/Invalid"])
    race = st.selectbox("race", ["Caucasian","AfricanAmerican","Asian","Hispanic","Other"])
    time_in_hosp = st.number_input("time_in_hospital (days)", 1, 100, 4)
    num_lab = st.number_input("num_lab_procedures", 0, 200, 40)
    num_proc = st.number_input("num_procedures", 0, 10, 1)
    num_meds = st.number_input("num_medications", 0, 100, 12)
    n_out = st.number_input("number_outpatient", 0, 50, 0)
    n_er = st.number_input("number_emergency", 0, 50, 0)
    n_in = st.number_input("number_inpatient", 0, 50, 0)
    n_dx = st.number_input("number_diagnoses", 1, 20, 5)

    adm_type_desc = st.selectbox("admission_type_desc", ["Emergency","Urgent","Elective","Newborn","Not Available","Trauma Center","Not Mapped"])
    disch_desc = st.selectbox("discharge_disposition_desc", ["Discharged to home","Discharged/transferred to SNF","Discharged/transferred to ICF","Left AMA","Expired","Hospice / home","Unknown/Invalid","Not Mapped"])
    adm_src_desc = st.selectbox("admission_source_desc", ["Physician Referral","Clinic Referral","HMO Referral","Transfer from a hospital","Emergency Room","Court/Law Enforcement","Not Available","Transfer from another health care facility","Unknown/Invalid","Not Mapped"])

    # diagnosis chapters
    diag1 = st.selectbox("diag_1_chapter", ["endocrine_metabolic","circulatory","respiratory","digestive","genitourinary","injury","ill_defined","supplemental_v","supplemental_e","other","unknown"])
    diag2 = st.selectbox("diag_2_chapter", ["endocrine_metabolic","circulatory","respiratory","digestive","genitourinary","injury","ill_defined","supplemental_v","supplemental_e","other","unknown"])
    diag3 = st.selectbox("diag_3_chapter", ["endocrine_metabolic","circulatory","respiratory","digestive","genitourinary","injury","ill_defined","supplemental_v","supplemental_e","other","unknown"])

    max_glu = st.selectbox("max_glu_serum", ["None","Norm",">200",">300"])
    a1c = st.selectbox("A1Cresult", ["None","Norm",">7",">8"])
    # diabetes meds columns use values: "No","Steady","Up","Down"
    insulin = st.selectbox("insulin", ["No","Steady","Up","Down"])
    metformin = st.selectbox("metformin", ["No","Steady","Up","Down"])
    glipizide = st.selectbox("glipizide", ["No","Steady","Up","Down"])
    glimepiride = st.selectbox("glimepiride", ["No","Steady","Up","Down"])
    glyburide = st.selectbox("glyburide", ["No","Steady","Up","Down"])
    change = st.selectbox("change", ["No","Ch"])
    diabetesMed = st.selectbox("diabetesMed", ["No","Yes"])

    row = {
        "race": race, "gender": gender, "age": age_bucket,
        "time_in_hospital": time_in_hosp, "num_lab_procedures": num_lab,
        "num_procedures": num_proc, "num_medications": num_meds,
        "number_outpatient": n_out, "number_emergency": n_er, "number_inpatient": n_in,
        "number_diagnoses": n_dx, "admission_type_desc": adm_type_desc,
        "discharge_disposition_desc": disch_desc, "admission_source_desc": adm_src_desc,
        "diag_1_chapter": diag1, "diag_2_chapter": diag2, "diag_3_chapter": diag3,
        "max_glu_serum": max_glu, "A1Cresult": a1c,
        "insulin": insulin, "metformin": metformin, "glipizide": glipizide,
        "glimepiride": glimepiride, "glyburide": glyburide,
        "change": change, "diabetesMed": diabetesMed
    }
    df_single = pd.DataFrame([row])

    def _ensure_model_columns(df_in: pd.DataFrame, model) -> pd.DataFrame:
        """Ensure the DataFrame contains all columns expected by the trained pipeline.
        Missing numeric columns get NaN (so the pipeline's SimpleImputer will fill them).
        Missing categorical columns get None.
        """
        try:
            pyfunc_impl = getattr(model, "_model_impl", None)
            if pyfunc_impl is None:
                return df_in
            python_model = getattr(pyfunc_impl, "python_model", None)
            if python_model is None:
                return df_in
            pipeline = getattr(python_model, "pipeline", None)
            if pipeline is None:
                return df_in
            pre = pipeline.named_steps.get("pre")
            if pre is None:
                return df_in

            expected_cols = []
            num_cols = []
            cat_cols = []
            # ColumnTransformer stores fitted transformers in .transformers_
            for name, transformer, cols in getattr(pre, "transformers_", []):
                if cols is None:
                    continue
                expected_cols.extend(cols)
                # heuristically assume the first transformer is numeric (name == 'num')
                if name == "num":
                    num_cols.extend(cols)
                elif name == "cat":
                    cat_cols.extend(cols)

            # Add missing columns with sensible placeholders
            for c in expected_cols:
                if c not in df_in.columns:
                    if c in num_cols:
                        df_in[c] = np.nan
                    else:
                        # use None so SimpleImputer can apply most_frequent
                        df_in[c] = None
        except Exception:
            # if anything unexpected happens, return df as-is and let the model raise
            return df_in
        # preserve column order by reindexing if possible
        try:
            return df_in.reindex(columns=expected_cols)
        except Exception:
            return df_in

    if st.button("Predict"):
        if st.session_state.model is None:
            st.warning("Load a model from the sidebar first.")
        else:
            df_safe = _ensure_model_columns(df_single.copy(), st.session_state.model)
            prob = st.session_state.model.predict(df_safe)[0]
            st.metric("30-day Readmission Probability", f"{prob:.2%}")
            st.progress(min(max(float(prob), 0.0), 1.0))

with tab2:
    st.subheader("Batch CSV")
    st.caption("Upload raw UCI-style CSV; the app expects the same columns as training (except 'readmitted').")
    file = st.file_uploader("CSV", type=["csv"])
    if file and st.session_state.model is not None:
        df = pd.read_csv(file)
        # Align columns expected by the model (add missing with NaN/None so the pipeline imputers can handle them)
        df_safe = _ensure_model_columns(df.copy(), st.session_state.model)
        preds = st.session_state.model.predict(df_safe)
        out = df.copy()
        out["readmission_prob"] = preds
        st.dataframe(out.head(50))
        st.download_button("Download predictions", out.to_csv(index=False).encode(), "readmission_predictions.csv", "text/csv")
    elif file and st.session_state.model is None:
        st.warning("Load a model first.")
