import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY environment variable is not set.")

client=Groq(api_key=api_key)
model="llama-3.3-70b-versatile"

knowledge_base={
    "age": "30",
    "sex": "male"
}

def retrieve_info(question):
    question=question.lower()
    if "age" in question:
        return knowledge_base["age"]
    elif "sex" in question:
        return knowledge_base["sex"]
    else:
        return None


def ask_llm(question):
    context=retrieve_info(question)

    sys_prompt=f"""Answer the question in one line only. Answer only based on this context. Do not hallucinate. Context: {context}"""
    system_prompt={
        "role": "system",
        "content": sys_prompt
    }

    message={
        "role": "user",
        "content": question
    }
    messages=[system_prompt, message]
    response = client.chat.completions.create(
    model=model,
    messages=messages
    )
    return response.choices[0].message.content

question="What is my sex?"
print(ask_llm(question))
