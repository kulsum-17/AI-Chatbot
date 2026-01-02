import streamlit as st
import joblib as jb
from groq import Groq
from dotenv import load_dotenv
import os
load_dotenv()
st.title("Academic Mentor")

st.write(
    "You can chat about your academics and also predict your marks "
    
)

st.sidebar.title("Features")

section = st.sidebar.selectbox(
    "Choose a feature",
    ["AI Chatbot", "Marks Predictor"]
)
 
model = jb.load("marks_model.pkl")
scaler = jb.load("scaler.pkl")

# == MARKS PREDICTOR 
if section == "Marks Predictor":
    st.header("📊 Marks Prediction")

    hours = st.number_input("Hours studied per day", 0.0, 12.0)
    prev_score = st.number_input("Previous score", 0.0, 100.0)
    extra = st.selectbox(
        "Extra curricular activity?",
        ["yes", "no"]
    )
    sleep = st.number_input("Sleep hours", 0.0, 9.0)
    q_paper = st.selectbox(
        "Previous year question practice (1–5)",
        [1, 2, 3, 4, 5]
    )

    extra_activity = 1 if extra == "yes" else 0

    if st.button("Predict my marks"):
        input_data = [[
            hours,
            prev_score,
            extra_activity,
            sleep,
            q_paper
        ]]

        #scaled_input = scaler.transform(input_data)
        pred = model.predict(input_data)[0]

       
        
        st.success(f"Your predicted marks are: {pred:.2f}")

#AI CHATBOT 
elif section == "AI Chatbot":
    st.header("💬 AI Chatbot for Academic Mentoring")
    
    client = Groq(api_key=st.secrets['GROQ_API_KEY'])

    user_input = st.text_area(
        "Ask any academic-related question"
    )

    if st.button("Get Answer") and user_input.strip():
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful academic mentor. "
                        "Answer only academic-related questions. "
                        "Encourage students to use the marks predictor "
                        "when relevant."
                    )
                },
                {
                    "role": "user",
                    "content": user_input
                }
            ]
        )

        st.write(
            "🧑‍🏫 :",
            response.choices[0].message.content
        )
