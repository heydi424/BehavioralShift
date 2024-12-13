import chainlit as cl
from typing import List
from ctransformers import AutoModelForCausalLM
import pickle
import numpy as np
from joblib import load

# Load the LLM
llm = AutoModelForCausalLM.from_pretrained(
    "zoltanctoth/orca_mini_3B-GGUF", model_file="orca-mini-3b.q4_0.gguf"
)

# Load the pre-trained Random Forest model
rf_model = load("/workspaces/BehavioralShift/random_forest_model.pk3")

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
    3: "Immediate intervention recommended."
}

# Interventions with empathetic and user-focused messages
interventions = {
    "alcohol": "Hi {name}, we understand that reducing alcohol use can feel challenging, "
               "but you're taking a brave first step. Here are some ways to work toward a healthier you:\n\n"
               "Alcohol Use: Social Replacement and Self-Control\n"
               "Goal: Replace drinking habits with healthier activities.\n"
               "Actions:\n"
               "- Identify triggers: Keep a daily log of situations or events that lead to drinking.\n"
               "- Substitute drinks: Replace alcohol with sparkling water or mocktails at gatherings.\n"
               "- Plan substance-free activities: Try hiking, painting, or movie nights.\n\n"
               "Resources:\n"
               "- [DrugFree](https://www.drugfree.org): Encourages teens to make healthy choices.\n"
               "- [Drinkaware: Drink Diary](https://www.drinkaware.co.uk/tools/drink-diary): Track alcohol consumption.\n"
               "- [National Institute on Alcohol Abuse and Alcoholism](https://www.niaaa.nih.gov): Learn about alcohol’s effects.\n",
    "marijuana": "Hi {name}, addressing marijuana use is a courageous step, and we're here to support you. "
                 "Consider these strategies to build healthier habits:\n\n"
                 "Marijuana Use: Building Healthier Coping Mechanisms\n"
                 "Goal: Reduce marijuana use by substituting it with stress-relieving activities.\n"
                 "Actions:\n"
                 "- Track usage: Write down when and why you use marijuana.\n"
                 "- Replace habits: Try journaling, yoga, or playing an instrument.\n\n"
                 "Resources:\n"
                 "- [My Life My Quit](https://mylifemyquit.com): Chat and text support for teens quitting marijuana.\n"
                 "- [NIDA for Teens](https://teens.drugabuse.gov): Reliable information on marijuana’s effects.\n",
    "cigarettes": "Hi {name}, quitting cigarettes is a big step, and you're showing incredible strength. "
                  "Here's how to get started:\n\n"
                  "Cigarette Use: Gradual Reduction and Self-Rewards\n"
                  "Goal: Reduce smoking by setting daily limits and rewarding progress.\n"
                  "Actions:\n"
                  "- Create a quit calendar: Reduce cigarettes per day (e.g., 10 to 7 in 1 week).\n"
                  "- Track cravings: Use a tracker to note the time and place of urges.\n\n"
                  "Resources:\n"
                  "- [Smokefree Teen](https://teen.smokefree.gov): Text support and tools for quitting smoking.\n",
    "combined": "Hi {name}, addressing the combined use of alcohol and marijuana takes courage, and we're here to help. "
                "Here’s how you can start:\n\n"
                "Combined Use (Alcohol + Marijuana): Balanced Reduction\n"
                "Goal: Address dual use by focusing on triggers and balanced reduction.\n"
                "Actions:\n"
                "- Monitor dual use: Keep a weekly log of combined use, noting triggers.\n"
                "- Plan alternatives: Replace one dual-use event with a recreational activity, like biking or painting.\n\n"
                "Resources:\n"
                "- [Dual Recovery Anonymous](https://draonline.org): Support for co-occurring substance use behaviors.\n",
    "general": "Hi {name}, you're taking an important step toward a better future, and we're here to support you. "
               "Here’s how you can start making positive changes:\n\n"
               "General Substance Use: Education and Self-Empowerment\n"
               "Goal: Empower yourself to make informed decisions about substance use.\n"
               "Actions:\n"
               "- Learn the facts: Watch educational videos on the effects of substances.\n"
               "- Explore purpose: Dedicate time to a skill or community activity, like volunteering or joining a club.\n\n"
               "Resources:\n"
               "- [NIDA for Teens](https://teens.drugabuse.gov): Accurate and engaging content on substance use.\n"
}

# Map the predicted risk levels to the new three levels
def map_risk_to_levels(predicted_risk: int) -> int:
    if predicted_risk in [1, 2]:  # Map original levels 1 and 2 to level 1
        return 1
    elif predicted_risk in [3]:  # Map original level 3 to level 2
        return 2
    elif predicted_risk in [4, 5]:  # Map original levels 4 and 5 to level 3
        return 3
    else:
        return 1  # Default to level 1 for safety

# Preprocess input data for the Random Forest model
def preprocess_for_model(response_history: List[str]) -> np.array:
    return np.array([int(resp) if resp.isdigit() else 0 for resp in response_history]).reshape(1, -1)

# Function to map risk and responses to specific substance intervention
def get_intervention(mapped_risk: int, user_responses: List[str]) -> str:
    if "alcohol" in user_responses:
        return interventions["alcohol"]
    elif "marijuana" in user_responses:
        return interventions["marijuana"]
    elif "cigarettes" in user_responses:
        return interventions["cigarettes"]
    elif "both" in user_responses:
        return interventions["combined"]
    else:
        return interventions["general"]

@cl.on_chat_start
async def on_chat_start():
    # Empathetic greeting and asking for the user's name
    cl.user_session.set("message_history", [])
    cl.user_session.set("question_index", 0)
    cl.user_session.set("user_responses", [])
    await cl.Message(content="Hello, welcome to BehavioralShift. I’m so proud of you for being here today. What’s your name?").send()

@cl.on_message
async def on_message(message: cl.Message):
    question_index = cl.user_session.get("question_index", 0)
    user_responses = cl.user_session.get("user_responses", [])
    message_history = cl.user_session.get("message_history", [])

    if question_index == 0:
        # Record user's name and start the questions
        user_name = message.content
        cl.user_session.set("user_name", user_name)
        await cl.Message(content=f"Thank you, {user_name}. Let's get started.").send()
        first_question = questions[0]
        await cl.Message(content=f"{first_question}").send()
        cl.user_session.set("question_index", 1)
    else:
        # Record user's response
        user_response = message.content
        user_responses.append(user_response)
        message_history.append(f"Q: {questions[question_index - 1]}\nA: {user_response}")

        # Move to the next question or conclude
        question_index += 1
        if question_index < len(questions) + 1:
            next_question = questions[question_index - 1]
            await cl.Message(content=f"{next_question}").send()
            cl.user_session.set("question_index", question_index)
            cl.user_session.set("user_responses", user_responses)
        else:
            # Preprocess responses and make a prediction
            model_input = preprocess_for_model(user_responses)
            predicted_risk = rf_model.predict(model_input)[0]
            mapped_risk = map_risk_to_levels(predicted_risk)

            # Get user's name and personalized intervention
            user_name = cl.user_session.get("user_name", "there")
            personalized_intervention = get_intervention(mapped_risk, user_responses).format(name=user_name)

            # Send the final assessment and intervention
            await cl.Message(
                content=(
                    f"Assessment complete, {user_name}.\n\n"
                    f"Predicted Risk Level: {mapped_risk}\n"
                    f"Intervention Level: {risk_levels[mapped_risk]}\n\n"
                    f"Personalized Intervention:\n{personalized_intervention}"
                )
            ).send()

            # Reset session
            cl.user_session.set("message_history", [])
            cl.user_session.set("question_index", 0)
            cl.user_session.set("user_responses", [])
