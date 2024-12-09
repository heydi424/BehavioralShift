import chainlit as cl
import openai
import pandas as pd
import joblib

# Load the Random Forest model
rf_model = joblib.load('random_forest_model.pkl')

# Set OpenAI API key
api_key = 'REMOVED'

openai.api_key = api_key

# Questions in a conversational format
questions = [
    "What is your age?",
    "What is your race or ethnicity?",
    "Are you Hispanic or Latino?",
    "What is your sex (Male/Female)?",
    "What grade are you in?",
    "How many times did you drink and drive in the last 30 days?",
    "At what age did you smoke your first whole cigarette?",
    "How many days have you smoked cigarettes in the past month?",
    "At what age did you have your first drink of alcohol?",
    "How many days have you consumed alcohol in the past month?",
    "How many times have you used weed in your lifetime?",
    "At what age did you first try weed?",
    "How many times have you used weed in the past month?",
    "How many times have you used cocaine?",
    "How many times have you used methamphetamine?",
    "Have you quit using tobacco in the last 12 months?",
    "How many cigarettes have you smoked in your lifetime?",
    "How many times have you used unprescribed drugs?"
]

# Encoding mappings for user responses
encoding_mappings = {
    # Include necessary mappings for all possible features
    "Yes": 1,
    "No": 0,
    # Add mappings for specific questions as needed
}

# ChatGPT interaction function
def chat_with_gpt(prompt):
    """
    Interact with ChatGPT to provide conversational feedback.
    """
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a behavioral intervention chatbot."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message["content"]

# Process user responses dynamically
def encode_response(question, response):
    """
    Encodes the user's response based on the expected feature mappings.
    """
    if question in encoding_mappings:
        return encoding_mappings.get(response, 0)  # Default to 0 if not found
    try:
        return int(response)  # Attempt to parse numeric responses
    except ValueError:
        return 0  # Default for non-numeric responses

@cl.on_chat_start
async def on_chat_start():
    """
    Start the Chainlit chatbot and prompt the user.
    """
    cl.user_session.set("responses", {})
    cl.user_session.set("current_question", 0)
    await cl.Message(content="Welcome to the Behavioral Intervention Chatbot! Let's start the conversation.").send()
    await ask_next_question()

async def ask_next_question():
    """
    Ask the next question in a conversational flow.
    """
    responses = cl.user_session.get("responses")
    current_index = cl.user_session.get("current_question")

    if current_index < len(questions):
        await cl.Message(content=questions[current_index]).send()
    else:
        await process_responses()

@cl.on_message
async def on_message(message: cl.Message):
    """
    Handle user responses to questions.
    """
    responses = cl.user_session.get("responses")
    current_index = cl.user_session.get("current_question")

    # Record the user's response
    current_question = questions[current_index]
    responses[current_question] = message.content
    cl.user_session.set("responses", responses)
    cl.user_session.set("current_question", current_index + 1)

    await ask_next_question()

async def process_responses():
    """
    Process user responses, predict risk level, and provide feedback.
    """
    responses = cl.user_session.get("responses")

    # Encode responses for the model
    encoded_responses = {
        question: encode_response(question, response)
        for question, response in responses.items()
    }

    # Convert responses to a DataFrame for the model
    user_data = pd.DataFrame([encoded_responses])

    # Align with model's expected features
    expected_features = rf_model.feature_names_in_
    for feature in expected_features:
        if feature not in user_data:
            user_data[feature] = 0  # Default for missing features
    user_data = user_data[expected_features].apply(pd.to_numeric, errors='coerce').fillna(0)

    # Predict risk level
    risk_level = rf_model.predict(user_data)[0]

    # Generate treatment plan based on risk level
    treatment_plan = f"Predicted Risk Level: {risk_level}\n\n"
    if risk_level == 1:
        treatment_plan += "Low Risk: Focus on educational resources and positive reinforcement."
    elif risk_level == 2:
        treatment_plan += "Moderate Risk: Monitor behavior and offer periodic counseling."
    elif risk_level == 3:
        treatment_plan += "High Risk: Recommend regular counseling and support groups."
    elif risk_level >= 4:
        treatment_plan += "Severe Risk: Immediate intervention and specialized treatment needed."

    # Use ChatGPT for a conversational summary
    summary_prompt = (
        f"The user's risk level is {risk_level}. Provide an empathetic and detailed explanation of their next steps."
    )
    conversational_response = chat_with_gpt(summary_prompt)

    # Send results to user
    await cl.Message(content=f"{treatment_plan}\n\nChatGPT Response:\n{conversational_response}").send()
    await cl.Message(content="Thank you for using the Behavioral Intervention Chatbot!").send()