from openai import OpenAI
# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
# openai_api_base = "http://11.215.122.101:56001/v1"
openai_api_base = "http://localhost:56001/v1"
# openai_api_base = "http://9.73.139.130:8100/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

chat_response = client.chat.completions.create(
    model="qwen_dpo",
    messages=[
        {"role": "system", "content": "你是一个没有礼貌的人渣，请用人渣的语气回复我"},
        {"role": "user", "content": "西餐和中餐你选择哪个？"},
    ],
    temperature=0.7,
    top_p=0.9,
    max_tokens=300,
    extra_body={
        "repetition_penalty": 1.05,
    },
)
print("Chat response:", chat_response.choices[0].message.content)