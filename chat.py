import chainlit as cl
from typing import List
from ctransformers import AutoModelForCausalLM


llm = AutoModelForCausalLM.from_pretrained(
       "zoltanctoth/orca_mini_3B-GGUF", model_file="orca-mini-3b.q4_0.gguf"
   )






def get_prompt(instruction: str, history: List[str]= None) -> str:
   system = "You are an AI assistant that gives helpful answers. You answer the questions in a short and concise way"
   prompt = f"### System:\n{system}\n\n### User:\n"
   if len(history)> 0:
       prompt += f"This is the conversation history: {''.join(history)}. Now answer the question:"
   prompt+= f"{instruction}\n\n### Response:\n"
   return prompt




@cl.on_message
async def on_message(message: cl.Message):
   message_history = cl.user_session.get("message_history")
   msg = cl.Message(content="")
   await msg.send()




   prompt = get_prompt(message.content, message_history)
   response = ""
   for word in llm(prompt, stream= True):
       await msg.stream_token(word)
       response += word
   await msg.update()
   message_history.append(response)




@cl.on_chat_start
def on_chat_start():
   cl.user_session.set("message_history", [])
   global llm
   llm = AutoModelForCausalLM.from_pretrained(
       "zoltanctoth/orca_mini_3B-GGUF", model_file="orca-mini-3b.q4_0.gguf"
   )


