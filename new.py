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

encoding_mappings = {
    'race/ethnicity': {'White': 0.5577019615134331,
  'Black or African American': 0.21749558427070745,
  'Multiple - Hispanic': 0.07249233057543925,
  'Multiple - Non-Hispanic': 0.054922376127173,
  'Asian': 0.046369805707911126,
  'Hispanic/Latino': 0.029060146881100677,
  'Am Indian / Alaska Native': 0.011378637166496236,
  'Native Hawaiian/other PI': 0.010579157757739146},
 'sex': {'Female': 0.5112577856279632, 'Male': 0.4887422143720368},
 'grade': {'9': 0.28370363484242817,
  '10': 0.2629915403923027,
  '11': 0.2409593752905085,
  '12': 0.2069164265129683,
  'Ungraded or other grade': 0.0054290229617923215},
 'Hispanic or Latino': {'No': 0.8969415264478944, 'Yes': 0.10305847355210561},
 'drink and drive in last 30 days': {'I did not drive the past 30 days': 0.5585386260109696,
  '0 times': 0.39706237798642746,
  '1 time': 0.01846239657897183,
  '2 or 3 times': 0.01044900994701125,
  '6 or more times': 0.01035604722506275,
  '4 or 5 times': 0.005131542251557125},
 'age of first whole cigarette': {'Never smoked a cigarette': 0.7611229896811379,
  '13 or 14 years old': 0.07855350004648136,
  '15 or 16 years old': 0.05869666263828205,
  '11 or 12 years old': 0.038802640141303336,
  '8 years old or younger': 0.03147717765176165,
  '9 or 10 years old': 0.02009854048526541,
  '17 years old': 0.011248489355768337},
 'days smoked cigarettes': {'0 days': 0.8709119643023148,
  '1 or 2 days': 0.03945337919494283,
  'All 30 days': 0.03302035883610672,
  '3 to 5 days': 0.018815654922376128,
  '10 to 19 days': 0.013461002138142605,
  '6 to 9 days': 0.012810263084503113,
  '20 to 29 days': 0.011527377521613832},
 'days used chewing tobacco': {'0 days': 0.9139537045644697,
  '1 or 2 days': 0.02900436924793158,
  'All 30 days': 0.018053360602398438,
  '3 to 5 days': 0.015524774565399274,
  '6 to 9 days': 0.010021381426048154,
  '10 to 19 days': 0.007976201543181185,
  '20 to 29 days': 0.005466208050571721},
 'days smoked cigars': {'0 days': 0.8735149205168727,
  '1 or 2 days': 0.054346007251092315,
  '3 to 5 days': 0.025825044157292927,
  'All 30 days': 0.015413219299061077,
  '6 to 9 days': 0.015376034210281678,
  '10 to 19 days': 0.010002788881658455,
  '20 to 29 days': 0.00552198568374082},
 'age of first drink': {'Never drank alcohol': 0.42532304545877103,
  '13 or 14 years old': 0.20174769917263177,
  '15 or 16 years old': 0.1582411453007344,
  '11 or 12 years old': 0.08407548573022218,
  '8 years old or younger': 0.06301013293669239,
  '9 or 10 years old': 0.04607232499767593,
  '17 years old': 0.021530166403272287},
 'days drinking alcohol': {'0 days': 0.6913265780422051,
  '1 or 2 days': 0.15647485358371294,
  '3 to 5 days': 0.07152551826717486,
  '6 to 9 days': 0.04088500511294971,
  '10 to 19 days': 0.021158315515478292,
  'All 30 days': 0.013386631960583806,
  '20 to 29 days': 0.005243097517895324},
 'how many days did you have 5+ drinks': {'0 days': 0.8210095751603607,
  '1 day': 0.06535279352979455,
  '2 days': 0.046314028074742025,
  '3 to 5 days': 0.0346007251092312,
  '6 to 9 days': 0.01591521799758297,
  '20 or more days': 0.010058566514827554,
  '10 to 19 days': 0.006749093613461002},
 'lifetime weed use': {'0 times': 0.6659477549502649,
  '100 or more times': 0.08143534442688483,
  '1 or 2 times': 0.07128381519010876,
  '3 to 9 times': 0.06825323045458771,
  '10 to 19 times': 0.041145300734405504,
  '20 to 39 times': 0.038077530910105045,
  '40 to 99 times': 0.03385702333364321},
 'age when first trying weed': {'Never tried marijuana': 0.661485544296737,
  '13 or 14 years old': 0.13156084410151528,
  '15 or 16 years old': 0.10863623686901552,
  '11 or 12 years old': 0.04488240215673515,
  '8 years old or younger': 0.021028167704750394,
  '9 or 10 years old': 0.01781165752533234,
  '17 years old': 0.014595147345914288},
 'times used weed': {'0 times': 0.812568560007437,
  '1 or 2 times': 0.06501812773077996,
  '3 to 9 times': 0.04744817328251371,
  '40 or more times': 0.033355024635121314,
  '10 to 19 times': 0.024579343683183045,
  '20 to 39 times': 0.017030770660964954},
 'cocaine usage': {'0 times': 0.9334200985404852,
  '1 or 2 times': 0.02617830250069722,
  '40 or more times': 0.015431811843450776,
  '3 to 9 times': 0.012642930184995817,
  '10 to 19 times': 0.007511387933438691,
  '20 to 39 times': 0.00481546899693223},
 'times huffing': {'0 times': 0.9000464813609742,
  '1 or 2 times': 0.04425025564748536,
  '3 to 9 times': 0.022589941433485174,
  '40 or more times': 0.015989588175141768,
  '10 to 19 times': 0.010765083201636143,
  '20 to 39 times': 0.006358650181277308},
 'heroin usage': {'0 times': 0.9503764990238914,
  '1 or 2 times': 0.015041368411267082,
  '40 or more times': 0.01308915125034861,
  '3 to 9 times': 0.010616342846518545,
  '10 to 19 times': 0.006563168169564005,
  '20 to 39 times': 0.004313470298410337},
 'meth usage': {'0 times': 0.9484428744073626,
  '1 or 2 times': 0.017588546992655944,
  '40 or more times': 0.01318211397229711,
  '3 to 9 times': 0.009928418704099657,
  '10 to 19 times': 0.0064887979920052056,
  '20 to 39 times': 0.004369247931579437},
 'ecstacy usage': {'0 times': 0.9157014037371014,
  '1 or 2 times': 0.03939760156177373,
  '3 to 9 times': 0.017867435158501442,
  '40 or more times': 0.013051966161569211,
  '10 to 19 times': 0.00892442130705587,
  '20 to 39 times': 0.005057172073998327},
 'steroid pills usage': {'0 times': 0.9487961327507669,
  '1 or 2 times': 0.016863437761457656,
  '40 or more times': 0.01238263456354002,
  '3 to 9 times': 0.011229896811378638,
  '10 to 19 times': 0.0065259830807846055,
  '20 to 39 times': 0.004201915032072139},
 'needle usage': {'0 times': 0.9606581760713954,
  '2 or more times': 0.0206563168169564,
  '1 time': 0.01868550711164823},
 'first time tobacco user in last 12 months': {'No': 0.8166403272287812,
  'Yes': 0.18335967277121873},
 'quit tobacco in last 12 months': {'I did not use tobacco products': 0.7299432927396114,
  'No, I did not completely quit': 0.16629171702147438,
  'Yes, I completely quit': 0.1037649902389142},
 'lifetime cigarette count': {'0 cigarettes': 0.697276192246909,
  'Never whole cigarette': 0.08240215673514921,
  '100 or more cigarettes': 0.06475783210932416,
  '2 to 5 cigarettes': 0.04912150227758669,
  '6 to 15 cigarettes': 0.030194292088872362,
  '26 to 99 cigarettes': 0.030157107000092963,
  '1 cigarette': 0.024412010783675747,
  '16 to 25 cigarettes': 0.021678906758389887},
 'unprescribed drugs usage': {'0 times': 0.9022775866877382,
  '1 or 2 times': 0.04387840475969136,
  '3 to 9 times': 0.02260853397787487,
  '40 or more times': 0.01308915125034861,
  '10 to 19 times': 0.011211304266988938,
  '20 to 39 times': 0.006935019057357999} }

encoding_mappings["age"] = "numeric"  # Age is treated as a float

# Map questions to corresponding encoding mappings keys
question_to_column = {
    "What is your race/ethnicity (e.g., White, Black, Asian)?": "race/ethnicity",
    "What is your age?": "age",
    "What is your sex (Male/Female)?": "sex",
    "What grade are you in (e.g., 9, 10, 11, 12)?": "grade",
    "Are you Hispanic or Latino (Yes/No)?": "Hispanic or Latino",
    "How many times did you drink and drive in the last 30 days?": "drink and drive in last 30 days",
    "At what age did you smoke your first whole cigarette?": "age of first whole cigarette",
    "How many days have you smoked cigarettes in the past month?": "days smoked cigarettes",
    "How many days have you used chewing tobacco in the past month?": "days used chewing tobacco",
    "How many days have you smoked cigars in the past month?": "days smoked cigars",
    "At what age did you have your first drink of alcohol?": "age of first drink",
    "How many days have you consumed alcohol in the past month?": "days drinking alcohol",
    "How many times did you have 5 or more drinks in the last 30 days?": "how many days did you have 5+ drinks",
    "How many times have you ever used weed in your lifetime?": "lifetime weed use",
    "At what age did you first try weed?": "age when first trying weed",
    "How many times have you used weed in the past month?": "times used weed",
    "How many times have you used cocaine?": "cocaine usage",
    "How many times have you inhaled substances to get high?": "times huffing",
    "How many times have you used heroin?": "heroin usage",
    "How many times have you used methamphetamine?": "meth usage",
    "How many times have you used ecstasy?": "ecstacy usage",
    "How many times have you used steroid pills or shots without a doctor's prescription?": "steroid pills usage",
    "How many times have you used needles to inject any illegal drug?": "needle usage",
    "Did you use tobacco for the first time in the last 12 months (Yes/No)?": "first time tobacco user in last 12 months",
    "Have you quit using tobacco in the last 12 months?": "quit tobacco in last 12 months",
    "How many cigarettes have you smoked in your lifetime?": "lifetime cigarette count",
    "How many times have you used unprescribed drugs?": "unprescribed drugs usage",
}

# Define the questions
questions = [
    "What is your race/ethnicity (e.g., White, Black, Asian)?",
    "What is your age?",
    "What is your sex (Male/Female)?",
    "What grade are you in (e.g., 9, 10, 11, 12)?",
    "Are you Hispanic or Latino (Yes/No)?",
    "How many times did you drink and drive in the last 30 days?",
    "At what age did you smoke your first whole cigarette?",
    "How many days have you smoked cigarettes in the past month?",
    "How many days have you used chewing tobacco in the past month?",
    "How many days have you smoked cigars in the past month?",
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
    "Did you use tobacco for the first time in the last 12 months (Yes/No)?",
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
    """
    Encodes user responses using `question_to_column` mapping.
    """
    encoded_responses = []
    invalid_inputs = []

    for question, user_response in zip(questions, response_history):
        col = question_to_column[question]

        if col == "age":
            try:
                # Directly convert age to float
                encoded_responses.append(float(user_response))
            except ValueError:
                # If conversion fails, set as -1
                encoded_responses.append(-1)
                invalid_inputs.append(f"Q: {col}, A: {user_response}")
        else:
            # Case-insensitive matching for encoded responses
            user_response_lower = user_response.strip().lower()
            encoded_value = next(
                (value for key, value in encoding_mappings[col].items()
                 if key.lower() == user_response_lower), -1
            )
            encoded_responses.append(encoded_value)
            if encoded_value == -1:
                invalid_inputs.append(f"Q: {col}, A: {user_response}")

    if invalid_inputs:
        print("Invalid Inputs Detected:", invalid_inputs)

    return np.array(encoded_responses).reshape(1, -1)




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
        user_name = message.content.strip()
        cl.user_session.set("user_name", user_name)
        await cl.Message(content=f"Thank you, {user_name}. Let's get started.").send()
        await cl.Message(content=questions[0]).send()
        cl.user_session.set("question_index", 1)
    else:
        user_response = message.content.strip()
        user_responses.append(user_response)
        message_history.append(f"Q: {questions[question_index - 1]}\nA: {user_response}")

        question_index += 1
        cl.user_session.set("question_index", question_index)
        cl.user_session.set("user_responses", user_responses)

        if question_index <= len(questions):
            next_question = questions[question_index - 1]
            await cl.Message(content=next_question).send()
        else:
            # Preprocess and check model input
            model_input = preprocess_for_model(user_responses)

            # Step 2: Check for Invalid Inputs
            if -1 in model_input:
                invalid_inputs = [
                    f"Q: {question_to_column[question]}, A: {resp}"
                    for question, resp in zip(questions, user_responses)
                    if isinstance(encoding_mappings[question_to_column[question]], dict) and
                    encoding_mappings[question_to_column[question]].get(resp.strip(), -1) == -1
                ]

                await cl.Message(
                    content=f"Invalid inputs detected: {invalid_inputs}. Please restart the process."
                ).send()
                return

            # Predict if all inputs are valid
            predicted_risk = rf_model.predict(model_input)[0]
            mapped_risk = map_risk_to_levels(predicted_risk)
            user_name = cl.user_session.get("user_name", "there")
            personalized_intervention = get_intervention(mapped_risk, user_responses).format(name=user_name)

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
