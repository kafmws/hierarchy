import os
import openai
import requests

os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''
os.environ['socks_proxy'] = ''
os.environ['all_proxy'] = ''

url = "http://172.25.11.252:5000/v1/chat/completions"

seed = 42

chat_template = """\n{%- for message in messages %}\n    {%- if message['role'] == 'system' -%}\n        {{- message['content'] + '\n\n' -}}\n    {%- else -%}\n        {%- if message['role'] == 'user' -%}\n            {{- name1 + ': ' + message['content'] + '\n'-}}\n        {%- else -%}\n            {{- name2 + ': ' + message['content'] + '\n' -}}\n        {%- endif -%}\n    {%- endif -%}\n{%- endfor -%}"""

chat_instruct_command = (
    """\nContinue the chat dialogue below. Write a single reply for the character "<|character|>".\n\n<|prompt|>"""
)

instruction_template_str = """\n{%- set ns = namespace(found=false) -%}\n{%- for message in messages -%}\n    {%- if message['role'] == 'system' -%}\n        {%- set ns.found = true -%}\n    {%- endif -%}\n{%- endfor -%}\n{%- if not ns.found -%}\n    {{- '[INST] <<SYS>>\n' + 'Answer the questions.' + '\n<</SYS>>\n\n' -}}\n{%- endif %}\n{%- for message in messages %}\n    {%- if message['role'] == 'system' -%}\n        {{- '[INST] <<SYS>>\n' + message['content'] + '\n<</SYS>>\n\n' -}}\n    {%- else -%}\n        {%- if message['role'] == 'user' -%}\n            {{-'' + message['content'] + ' [/INST] '-}}\n        {%- else -%}\n            {{-'' + message['content'] + ' </s><s>[INST] ' -}}\n        {%- endif -%}\n    {%- endif -%}\n{%- endfor -%}\n{%- if add_generation_prompt -%}\n    {{-''-}}\n{%- endif -%}"""

body = {
    "model": "gpt-3.5-turbo",
    "frequency_penalty": 0,
    "logit_bias": {},
    "max_tokens": 0,
    "n": 1,
    "presence_penalty": 0,
    "stop": "string",
    "stream": False,
    "temperature": 1,
    "top_p": 1,
    "user": "kafm",
    "mode": "instruct",
    "instruction_template": "Llama-v2",
    "instruction_template_str": instruction_template_str,
    "character": "Assistant",
    "name1": "You",
    "name2": "AI",
    "context": "The following is a conversation with an AI Large Language Model. The AI has been trained to answer questions, provide recommendations, and help with decision making. The AI follows user requests. The AI thinks outside the box.",
    "greeting": "How can I help you today?",
    "chat_template_str": chat_template,
    "chat_instruct_command": chat_instruct_command,
    "continue_": False,
    "preset": "simple-1",
    "min_p": 0,
    "dynamic_temperature": False,
    "dynatemp_low": 1,
    "dynatemp_high": 1,
    "dynatemp_exponent": 1,
    "top_k": 0,
    "repetition_penalty": 1,
    "repetition_penalty_range": 1024,
    "typical_p": 1,
    "tfs": 1,
    "top_a": 0,
    "epsilon_cutoff": 0,
    "eta_cutoff": 0,
    "guidance_scale": 1,
    "negative_prompt": "",
    "penalty_alpha": 0,
    "mirostat_mode": 0,
    "mirostat_tau": 5,
    "mirostat_eta": 0.1,
    "temperature_last": False,
    "do_sample": True,
    "seed": seed,
    "encoder_repetition_penalty": 1,
    "no_repeat_ngram_size": 0,
    "min_length": 0,
    "num_beams": 1,
    "length_penalty": 1,
    "early_stopping": False,
    "truncation_length": 0,
    "max_tokens_second": 0,
    "prompt_lookup_num_tokens": 0,
    "custom_token_bans": "",
    "auto_max_new_tokens": False,
    "ban_eos_token": False,
    "add_bos_token": True,
    "skip_special_tokens": True,
    "grammar_string": "",
}

history = []

headers = {"Content-Type": "application/json"}


def api_test():
    print("input `exit()` for exit")
    user_message = ''

    while True and user_message != 'exit()':
        user_message = input(">")
        history.append({"role": "system", "content": "You are a helpful assistant."})
        history.append({"role": "user", "content": user_message})
        body['messages'] = history

        # print(body)
        response = requests.post(url, headers=headers, json=body, verify=False)
        status = response.status_code
        # print(status)
        res = response.json()
        # print(res)
        assistant_message = res['choices'][0]['message']['content']
        history.append({"role": "assistant", "content": assistant_message})
        print(assistant_message)


def openai_impl():
    os.environ['OPENAI_API_KEY'] = ''
    # os.environ['OPENAI_BASE_URL'] = 'http://172.25.11.252:5000'
    os.environ['OPENAI_API_HOST'] = 'http://127.0.0.1:5000'
    openai.api_key = ''
    openai.api_base = 'http://172.25.11.252:5000/v1'
    openai.api_version = '2023-05-15'

    client = openai.OpenAI()

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Who won the world series in 2020?"},
            {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
            {"role": "user", "content": "Where was it played?"},
        ],
    )

    print(response)


if __name__ == '__main__':
    # api_test()
    openai_impl()
