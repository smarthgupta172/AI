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
role="user"

from pydantic import BaseModel
class Ticket(BaseModel):
    name:str
    email:str
    issue:str

schema=Ticket.model_json_schema()

response_format={
    "type": "json_object"
}

system_prompt=f"""
Extract the personal information from the ticket strictly based on this schema and give a json output.
{schema}
"""

message_system={
    "role": "system",
    "content": system_prompt
}

text="Hello my name is Smarth. My phone got stolen and I want to track it down. Can you help me with that? My phone is an iPhone 14 pro max and the IMEI number is 123456789012345. Please provide me with the steps to track it down.My email address is abs@gmail.com.My phone number is 1234567890. Please help me out."
prompt= text + "\n\nPlease extract the personal information from the above text and provide it in a json format strictly based on the schema provided."
# message me role and content
message={
    "role": role,
    "content": prompt
}

messages=[message_system, message]

response=client.chat.completions.create(model=model, messages=messages,response_format=response_format)

answer=response.choices[0].message.content
print(answer)