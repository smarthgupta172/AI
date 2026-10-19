import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
my_api_key=os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("API key kaha hai bhai")

client=Groq(api_key=my_api_key)
model="llama-3.3-70b-versatile"


def llm_ans(prompt):
    message={
        "role":"user",
        "content": prompt
    }
    messages=[message]
    response=client.chat.completions.create(model=model, messages=messages)
    ans=response.choices[0].message.content
    return ans


prompt = """
Classify the complaint into exactly one of these categories:
billing
technical
return

If it doesn't fit, answer OTHER.

Reply with only one word.

Complaint:
My laptop stopped working.
"""

print(llm_ans(prompt))