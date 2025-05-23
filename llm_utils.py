import os
import openai
import json

OPENAPI_MODEL = "gpt-4o-2024-08-06"

def query_llm(prompt):
    try:
        openai.api_key = os.environ.get("OPENAI_API_KEY")
        response = openai.chat.completions.create(
            model=OPENAPI_MODEL,
            messages=[
                {"role": "system", "content": "You are a data matching assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
        )
        reply = response.choices[0].message.content
        return json.loads(reply)
    except Exception as e:
        print("⚠️ LLM Error:", e)
        return {
            "Organization": "N/A",
            "Sectors": "N/A",
            "Factor Id": "N/A",
            "Factor Name": "N/A"
        }
