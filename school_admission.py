import pandas as pd
import numpy as np
import streamlit as st
import qrcode
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Page Configuration
st.set_page_config(page_title="The Perfect School Portal", page_icon="🏫", layout="wide")

# Initialize session state for database reloads to prevent infinite refresh freezes
if 'data_version' not in st.session_state:
    st.session_state['data_version'] = 0

# 2. Safely Load data file 
try:
    df = pd.read_csv("school_student_performance.csv")
except Exception as e:
    st.error(f"Data File Missing! Ensure 'school_student_performance.csv' is present in the main directory. Error: {e}")
    st.stop()

# 3. Structural Encoding Pipelines
X_raw = df[['Class_Attendance_Percent', 'Monthly_Test_Score', 'Practice_Hours_Per_Week', 'Smart_Board_Learning', 'Teaching_Methodology']]
X_encoded = pd.get_dummies(X_raw, columns=['Smart_Board_Learning', 'Teaching_Methodology'], drop_first=True, dtype=int)
y = df['Final_Performance_Percentage']

# Ensure strict consistency across trained features
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# 4. Supervised Training Phase
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Evaluate System Matrices
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

# --- WEB PORTAL RENDERING VISUAL INTERFACE ---
st.title("🏫 AI-Powered The Perfect School Enrollment & Performance Portal")
st.write("The Perfect School — Performance evaluation using advanced interactive parameters.")
st.info("Supervised Regression predicts a Number. In this project, our target is Final_Performance_Percentage.")

logs_tab, evaluate_tab = st.tabs(["📋 View 90 Student Logs", "🎯 Supervised Performance Predictor & QR"])

with logs_tab:
    st.subheader("Historical Student Records & Incentive Tracking (Last 2 Months)")
    selected_class = st.selectbox("Choose Target Class to Inspect Records:", ["All Classes", "Nursery", "1st Class", "2nd Class", "3rd Class", "4th Class", "5th Class"])
    
    filtered_data = df if selected_class == "All Classes" else df[df["Class_Level"] == selected_class]
    st.dataframe(filtered_data, use_container_width=True, hide_index=True)
    
    st.divider()
    l_col, m_col, r_col = st.columns(3)
    l_col.metric("Total Profiled Records", len(df))
    l_col.caption(f"Database Current Version Key: Logs-{st.session_state['data_version']}")
    m_col.metric("Model Evaluation Error (MAE)", f"{mae:.2f} marks")
    r_col.metric("Model Goodness of Fit (R2 Score)", f"{r2:.3f}")

with evaluate_tab:
    st.subheader("Evaluate Custom Student Learning Scenario")
    st.write("Enter parameters below to run the supervised linear regression algorithm array.")
    
    with st.form("advanced_evaluation_form"):
        left_c, right_col = st.columns(2)
        with left_c:
            c_lvl = st.selectbox("Current Student Level:", ["Nursery", "1st Class", "2nd Class", "3rd Class", "4th Class", "5th Class"])
            attendance_val = st.slider("Class Attendance Rate (%)", 65, 100, 85)
            test_val = st.slider("Average Assessment Marks (0-100)", 45, 98, 75)
        with right_col:
            hours_val = st.slider("Practical/Hands-on Learning Hours Per Week", 4, 22, 12)
            smart_board = st.selectbox("Exposure to Smart Board Interactive Learning?", ["Yes", "No"])
            methodology = st.selectbox("Primary Teaching Methodology Profile:", ["Activity-Based Hands-on", "Traditional Lecture"])
            
        process_btn = st.form_submit_button("Run Analytics Engine", type="primary", use_container_width=True)
        
    if process_btn:
        smart_board_yes = 1 if smart_board == "Yes" else 0
        method_traditional = 1 if methodology == "Traditional Lecture" else 0
        
        test_vector = pd.DataFrame(
            [[attendance_val, test_val, hours_val, smart_board_yes, method_traditional]], 
            columns=['Class_Attendance_Percent', 'Monthly_Test_Score', 'Practice_Hours_Per_Week', 'Smart_Board_Learning_Yes', 'Teaching_Methodology_Traditional Lecture']
        )
        
        modeled_score = float(model.predict(test_vector))
        modeled_score = max(40, min(100, modeled_score)) # Lock boundaries naturally
        
        st.metric("Estimated Final Term Score Percentage", f"{modeled_score:.1f}%")
        
        if modeled_score >= 60:
            st.balloons()
            st.success("Verification Status: Satisfactory Academic Progression Confirmed.")
            
            qr_data_string = (f"Class: {c_lvl}\n"
                              f"Method: {methodology}\n"
                              f"Smart Board: {smart_board}\n"
                              f"Calculated Score: {modeled_score:.1f}%\n"
                              f"Discount Log: 25% Off Scholarship Eligible")
            
            qr_img = qrcode.make(qr_data_string)
            stream_buf = BytesIO()
            qr_img.save(stream_buf, format="PNG")
            st.image(stream_buf.getvalue(), caption=f"Secure System QR Code generated for {c_lvl} record verification.", width=210)
        else:
            st.warning("Performance parameters fall below progression standards. Target profile requires mandatory counseling.")

# --- SIDEBAR DATABASE APPEND FORM LOGS ---
st.sidebar.markdown("---")
st.sidebar.subheader("📝 Admin Live Data Entry")

with st.sidebar.form("admin_entry_form", clear_on_submit=True):
    new_roll = st.text_input("Student Roll No:")
    new_class = st.selectbox("Class:", ["Nursery", "1st Class", "2nd Class", "3rd Class", "4th Class", "5th Class"])
    new_att = st.number_input("Attendance %:", 0, 100, 85)
    new_test = st.number_input("Test Marks (0-100):", 0, 100, 75)
    new_hours = st.number_input("Practice Hours/Week:", 0, 30, 10)
    new_sb = st.selectbox("Smart Board?", ["Yes", "No"])
    new_tech = st.selectbox("Methodology:", ["Activity-Based", "Traditional Lecture"])
    new_inc = st.selectbox("Package:", ["Standard Package", "Free Uniform + Activity Box"])
    new_disc = st.selectbox("Discount:", ["0% Off", "25% Off"])
    new_final = st.number_input("Final Term Percentage:", 0, 100, 70)
    
    admin_submit = st.form_submit_button("Save Student to Database")

if admin_submit and new_roll:
    new_row = pd.DataFrame([[new_roll, new_class, new_att, new_test, new_hours, new_sb, new_tech, new_inc, new_disc, new_final]], columns=df.columns)
    new_row.to_csv("school_student_performance.csv", mode='a', header=False, index=False)
    
    # Safe Trigger: Increment layout memory keys instead of brute force endless rerun loops
    st.session_state['data_version'] += 1
    st.sidebar.success(f"Record for {new_roll} permanently saved to database logs!")
    #st.rerun()
