import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import gradio as gr
import pandas as pd
import requests
import torch
from dotenv import load_dotenv
from llmlingua import PromptCompressor

from utils import activate_button, create_metrics_df, update_label

load_dotenv()

LLM_ENDPOINT = os.getenv("LLM_ENDPOINT")
LLM_MODELS = ["meta-llama/Meta-Llama-3-70B-Instruct", "mistral-7b-q4", "CohereForAI/c4ai-command-r-plus"]

llm_lingua = PromptCompressor(
    model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
    use_llmlingua2=True,
    device_map="mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu",
)


def call_llm_api(prompt: str, model: str):
    headers = {"Content-Type": "application/json", "Authorization": "Bearer no-key"}
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1000,
    }
    response = requests.post(LLM_ENDPOINT, headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        response_obj = response.json()
        return response_obj["choices"][0]["message"]["content"], response_obj
    else:
        return response.text, f"Error for model {model}: {response.status_code} - {response.text}"


def compress_prompt(prompt: str, rate: float):
    result = llm_lingua.compress_prompt(prompt, rate=rate)

    return result["compressed_prompt"], create_metrics_df(result)


def run(prompt: str, rate: float, target_model: str):
    with ThreadPoolExecutor() as executor:
        start = time.time()
        future_response_original = executor.submit(call_llm_api, prompt, target_model)
        compressed_prompt, metrics = compress_prompt(prompt, rate)
        res_compressed, res_compressed_obj = call_llm_api(compressed_prompt, target_model)
        res_original, res_original_obj = future_response_original.result()
        print(f"Processing time: {time.time() - start:.2f}s")

    return (
        compressed_prompt,
        metrics,
        res_compressed,
        res_compressed_obj,
        res_original,
        res_original_obj,
    )


flagging_callback = gr.CSVLogger()

with gr.Blocks() as demo:
    gr.Markdown(
        """
        # Prompt Compression A/B Test
        Your prompt is sent to a target LLM model for completion, once uncompressed and once compressed using LLMLingua-2. Compare the responses and select the better one.
        Note: the order of the responses is random.
    """
    )
    prompt = gr.Textbox(lines=8, label="Prompt")
    rate = gr.Slider(0.1, 1, 0.5, step=0.05, label="Rate")
    target_model = gr.Radio(LLM_MODELS, value=LLM_MODELS[0], label="Target LLM Model")
    with gr.Row():
        clear = gr.Button("Clear", variant="secondary")
        submit = gr.Button("Submit", variant="primary", interactive=False)

    compressed_prompt = gr.Textbox(label="Compressed Prompt", visible=False)
    metrics = gr.Dataframe(
        headers=[*create_metrics_df().columns.values],
        row_count=1,
        height=90,
        label="Metrics",
        show_label=False,
        interactive=False,
    )
    with gr.Row(variant="compact"):
        with gr.Column():
            response_a = gr.Textbox(label="LLM Response A", lines=10, max_lines=10, autoscroll=False, interactive=False)
            response_a_obj = gr.Textbox(label="Response A", visible=False)
            button_a = gr.Button("A is better", interactive=False)
        with gr.Column():
            response_b = gr.Textbox(label="LLM Response B", lines=10, max_lines=10, autoscroll=False, interactive=False)
            response_b_obj = gr.Textbox(label="Response B", visible=False)
            button_b = gr.Button("B is better", interactive=False)

    prompt.change(activate_button, inputs=prompt, outputs=submit)
    submit.click(
        run,
        inputs=[prompt, rate, target_model],
        outputs=[compressed_prompt, metrics, response_a, response_a_obj, response_b, response_b_obj],
    )
    clear.click(
        lambda: [create_metrics_df()] + [None] * 6 + [0.5],
        outputs=[
            metrics,
            prompt,
            compressed_prompt,
            response_a_obj,
            response_a,
            response_b_obj,
            response_b,
            rate,
        ],
    )

    response_a.change(lambda x: update_label(x, response_a), inputs=response_a, outputs=response_a)
    response_b.change(lambda x: update_label(x, response_b), inputs=response_b, outputs=response_b)

    def flag(flag_value: str, *args):
        flagging_callback.flag(*args, flag_value)
        return [activate_button(False)] * 2

    FLAG_COMPONENTS = [prompt, compressed_prompt, rate, metrics, response_a_obj, response_b_obj]
    response_a.change(activate_button, inputs=response_a, outputs=button_a)
    response_b.change(activate_button, inputs=response_b, outputs=button_b)
    flagging_callback.setup(FLAG_COMPONENTS, "flagged")
    button_a.click(
        lambda *args: flag("A", args), inputs=FLAG_COMPONENTS, outputs=[button_a, button_b], preprocess=False
    )
    button_b.click(
        lambda *args: flag("B", args), inputs=FLAG_COMPONENTS, outputs=[button_a, button_b], preprocess=False
    )

    examples = gr.Examples(
        examples=[
            [
                "John: So, um, I've been thinking about the project, you know, and I believe we need to, uh, make some changes. I mean, we want the project to succeed, right? So, like, I think we should consider maybe revising the timeline. Sarah: I totally agree, John. I mean, we have to be realistic, you know. The timeline is, like, too tight. You know what I mean? We should definitely extend it.",
                0.3,
            ],
            [
                "Item 15, report from City Manager Recommendation to adopt three resolutions. First, to join the Victory Pace program. Second, to join the California first program. And number three, consenting to to inclusion of certain properties within the jurisdiction in the California Hero program. It was emotion, motion, a second and public comment. CNN. Please cast your vote. Oh. Was your public comment? Yeah. Please come forward. I thank you, Mr. Mayor. Thank you. Members of the council. My name is Alex Mitchell. I represent the hero program. Just wanted to let you know that the hero program. Has been in California for the last three and a half years. We’re in. Over 20. We’re in 28 counties, and we’ve completed over 29,000 energy efficient projects to make homes. Greener and more energy efficient. And this includes anything. From solar to water. Efficiency. We’ve done. Almost.$550 million in home improvements.",
                0.5,
            ],
        ],
        inputs=[prompt, rate],
    )


demo.launch()
