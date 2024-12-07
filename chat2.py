import chainlit as cl
from typing import List
from ctransformers import AutoModelForCausalLM

# Load the LLM
llm = AutoModelForCausalLM.from_pretrained(
    "zoltanctoth/orca_mini_3B-GGUF", model_file="orca-mini-3b.q4_0.gguf"
)

# Define the questions
questions = [
    "What is your age?",
    "What is your race/ethnicity (e.g., White, Black, Asian)?",
    "Are you Hispanic or Latino (Yes/No)?",
    "What is your sex (Male/Female)?",
    "What grade are you in (e.g., 9, 10, 11, 12)?",
    "How many times did you drink and drive in the last 30 days?",
    "At what age did you smoke your first whole cigarette?",
    "How many days have you smoked cigarettes in the past month?",
    "How many days have you used chewing tobacco in the past month?",
    "How many days have you smoked cigars in the past month?",
    "Did you use tobacco for the first time in the last 12 months (Yes/No)?",
    "At what age did you have your first drink of alcohol?",
    "How many days have you consumed alcohol in the past month?",
    "How many times did you have 5 or more drinks in the last 30 days?",
    "How many times have you ever used weed in your lifetime?",
    "At what age did you first try weed?",
    "How many times have you used weed in the past month?",
    "How many times have you used cocaine?",
    "How many times have you inhaled substances to get high?",
    "How many times have you used heroin?",
    "How many times have you used methamphetamine?",
    "How many times have you used ecstasy?",
    "How many times have you used steroid pills or shots without a doctor's prescription?",
    "How many times have you used needles to inject any illegal drug?",
    "Have you quit using tobacco in the last 12 months?",
    "How many cigarettes have you smoked in your lifetime?",
    "How many times have you used unprescribed drugs?"
]

# Risk level thresholds
risk_levels = {
    1: "No intervention needed.",
    2: "Monitor periodically.",
    3: "Provide brief counseling.",
    4: "Recommend specialized intervention.",
    5: "Immediate and intensive intervention required."
}

# Function to calculate risk level
def calculate_risk_level(total_score: int) -> str:
    if total_score <= 5:
        return risk_levels[1]
    elif total_score <= 10:
        return risk_levels[2]
    elif total_score <= 15:
        return risk_levels[3]
    elif total_score <= 20:
        return risk_levels[4]
    else:
        return risk_levels[5]

@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("message_history", [])
    cl.user_session.set("question_index", 0)
    cl.user_session.set("total_score", 0)
    first_question = questions[0]
    await cl.Message(content=f"Welcome to BehavioralShift. Let's begin.\n\n{first_question}").send()

@cl.on_message
async def on_message(message: cl.Message):
    question_index = cl.user_session.get("question_index", 0)
    total_score = cl.user_session.get("total_score", 0)
    message_history = cl.user_session.get("message_history", [])

    # Record user's response
    user_response = message.content
    message_history.append(f"Q: {questions[question_index]}\nA: {user_response}")

    # Simulate scoring logic (replace with real logic as needed)
    try:
        score = int(user_response) if user_response.isdigit() else 1
    except ValueError:
        score = 1  # Default score for non-numeric responses
    total_score += score

    # Move to the next question or conclude
    question_index += 1
    if question_index < len(questions):
        next_question = questions[question_index]
        await cl.Message(content=f"{next_question}").send()
        cl.user_session.set("question_index", question_index)
        cl.user_session.set("total_score", total_score)
    else:
        # Calculate risk level and conclude
        risk_level = calculate_risk_level(total_score)
        await cl.Message(
            content=f"Assessment complete.\n\nTotal Score: {total_score}\nIntervention Level: {risk_level}"
        ).send()
        cl.user_session.set("message_history", [])
        cl.user_session.set("question_index", 0)
        cl.user_session.set("total_score", 0)

