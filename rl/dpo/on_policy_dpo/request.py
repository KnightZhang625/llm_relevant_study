# coding:utf-8

from openai import OpenAI
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:56001/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

chat_response = client.chat.completions.create(
    model="llama-8b",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"},
    ],
    temperature=0.7,
    top_p=0.9,
    max_tokens=300,
    extra_body={
        "repetition_penalty": 1.05,
    },
)
print("Chat response:", chat_response.choices[0].message.content)