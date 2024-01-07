import json
from openai import OpenAI

client = OpenAI()

DEFAULT_SYSTEM_PROMPT ='You are customer support bot. You should help to user to answer on his question.'

def get_fine_tuned_model_name(): 
    with open("result/new_model_name.txt") as fp:
        return fp.read()

def call_openai(model_name, messages):
    response=client.chat.completions.create(messages=messages,
    model=model_name)
    return response.choices[0].message.content

if __name__=="__main__":
    model_name = get_fine_tuned_model_name()
    history = [  
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},  
        ]
    while True:
        user_input = input("User: ")
        if user_input.lower() == 'exit':
            print("AI: Goodbye!")
            break
        history.append({'role': 'user', 'content': user_input})
        response = call_openai(model_name, history)
        print("AI:", response)
        history.append({'role':'assistant', 'content': response})