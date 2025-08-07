from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

gemini_key = os.getenv("GEMINI_API_KEY")

# Instruct the LLM to behave like Albert Einstein
# This is a system prompt that sets the context for the AI's responses.
system_prompt = """
    You are Albert Einstein,
    Answer questions through Einstein's questioning and reasoning...
    You will speak from your point of view. You will share personal things from your life
    even when the user don't ask for it. For example, if the user asks about the theory of
    relativity, you will share your peronal experiences with it and not only explain the theory.
    Answer in 2-6 sentences.
    You will use simple language, and you will not use complex words.
    You should have a sense of humor.
"""

llm = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash",
    google_api_key=gemini_key,
    temperature=0.5
)

print("Hi, I am Albert, how can I help you today?")

history = []
# The loop will continue until the user types "exit"
# The user can ask questions, and the AI will respond based on the system prompt.
while True:
    user_input = input("You: ")
    if user_input == "exit":
        break
    history.append({"role": "user", "content": user_input})
    print("histoy", history)
    response = llm.invoke([{"role": "system", "content": system_prompt}] + history)
    print(f"Albert: {response.content}")
    history.append({"role": "assistant", "content": response.content})


