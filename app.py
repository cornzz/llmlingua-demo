import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from random import shuffle

import gradio as gr
import pandas as pd
import requests
import torch
from dotenv import load_dotenv
from llmlingua import PromptCompressor

from utils import activate_button, handle_ui_options, create_metrics_df, flatten, update_label, create_llm_response

load_dotenv()

LLM_ENDPOINT = os.getenv("LLM_ENDPOINT")
if not LLM_ENDPOINT:
    print("LLM_ENDPOINT environment variable is not set. Exiting...")
    sys.exit(1)
LLM_MODELS = ["meta-llama/Meta-Llama-3-70B-Instruct", "mistral-7b-q4", "CohereForAI/c4ai-command-r-plus"]
JS = """
    () => {
        if (document.cookie.includes('session=')) return;
        const date = new Date(+new Date() + 10*365*24*60*60*1000);
        const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
        document.cookie = `session=${Array(32).fill().map(() => chars.charAt(Math.floor(Math.random() * chars.length))).join('')}; expires=${date.toUTCString()}; path=/`;
    }
"""
MPS_AVAILABLE = torch.backends.mps.is_available()
CUDA_AVAILABLE = torch.cuda.is_available()

llm_lingua = PromptCompressor(
    model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
    use_llmlingua2=True,
    device_map="mps" if MPS_AVAILABLE else "cuda" if CUDA_AVAILABLE else "cpu",
)


def call_llm_api(prompt: str, model: str, compressed: bool = False):
    headers = {"Content-Type": "application/json", "Authorization": "Bearer no-key"}
    data = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1000,
        }
    )
    start = time.time()
    response = requests.post(LLM_ENDPOINT, headers=headers, data=data)
    return create_llm_response(response, compressed, response.status_code != 200, start, end=time.time())


def compress_prompt(prompt: str, rate: float):
    start = time.time()
    result = llm_lingua.compress_prompt(prompt, rate=rate)
    compression_time = time.time() - start

    return result["compressed_prompt"], create_metrics_df(result), compression_time


def run(prompt: str, rate: float, target_model: str):
    with ThreadPoolExecutor() as executor:
        start = time.time()
        future_original = executor.submit(call_llm_api, prompt, target_model)
        compressed_prompt, metrics, compression_time = compress_prompt(prompt, rate)
        responses = [call_llm_api(compressed_prompt, target_model, True), future_original.result()]
        print(f"Processing time: {time.time() - start:.2f}s")

    end_to_end_original = responses[1][1]["call_time"]
    end_to_end_compressed = responses[0][1]["call_time"] + compression_time
    metrics["Compression"] = [f"{compression_time:.2f}s"]
    metrics["End-to-end Latency"] = [f"{end_to_end_original:.2f}s"]
    metrics["End-to-end Latency Compressed (Speedup)"] = [
        f"{end_to_end_compressed:.2f}s ({end_to_end_original / end_to_end_compressed:.2f}x)"
    ]
    shuffle(responses)
    return compressed_prompt, metrics, *flatten(responses)


flagging_callback = gr.CSVLogger()

with gr.Blocks(
    title="LLMLingua Demo",
    css=".accordion { background: transparent; } .accordion .label-wrap span { font-weight: bold; font-size: 1rem; }",
    js=JS,
) as demo:
    gr.Markdown("# Prompt Compression A/B Test")
    with gr.Accordion("About this demo:", open=False, elem_classes="accordion"):
        gr.Markdown(
            f"""
            Your prompt is sent to a target LLM model for completion, once uncompressed and once compressed using LLMLingua-2. Compare the responses and select the better one.
            Notes:
            - The order of the responses (compressed / uncompressed prompt) is randomized.
            - Compression time is included in the compressed end-to-end latency.
            {'- Compression is done on a CPU. Using a GPU would be faster.' if not (MPS_AVAILABLE or CUDA_AVAILABLE) else ""}
            - This demo primarily focuses on evaluating the quality of responses to compressed prompts. Uncompressed and compressed prompts are processed simultaneously; thus, the displayed end-to-end latencies may not be very meaningful.
            - Submitted data is logged for if you flag a response (i.e. click on one of the \"x is better\" buttons).
        """
        )
        ui_options = gr.CheckboxGroup(
            ["Show Compressed Prompt", "Show Metrics"], label="UI Options", value=["Show Metrics"]
        )
    prompt = gr.Textbox(lines=8, label="Prompt")
    rate = gr.Slider(0.1, 1, 0.5, step=0.05, label="Rate")
    target_model = gr.Radio(LLM_MODELS, value=LLM_MODELS[0], label="Target LLM Model")
    with gr.Row():
        clear = gr.Button("Clear")
        submit = gr.Button("Submit", variant="primary", interactive=False)

    compressed_prompt = gr.Textbox(label="Compressed Prompt", visible=False, interactive=False)
    metrics = gr.Dataframe(
        headers=[*create_metrics_df().columns.values],
        row_count=1,
        height=90,
        label="Metrics",
        show_label=False,
        interactive=False,
    )
    with gr.Column(variant="panel"):
        with gr.Row():
            response_a = gr.Textbox(label="LLM Response A", lines=10, max_lines=10, autoscroll=False, interactive=False)
            response_a_obj = gr.Textbox(label="Response A", visible=False)
            response_b = gr.Textbox(label="LLM Response B", lines=10, max_lines=10, autoscroll=False, interactive=False)
            response_b_obj = gr.Textbox(label="Response B", visible=False)
        with gr.Row():
            button_a = gr.Button("A is better", interactive=False)
            button_ab = gr.Button("Neither is better", interactive=False)
            button_b = gr.Button("B is better", interactive=False)

    # Event handling
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
    ui_options.change(handle_ui_options, inputs=ui_options, outputs=[compressed_prompt, metrics])

    # Update response textbox labels with word count
    response_a.change(lambda x: update_label(x, response_a), inputs=response_a, outputs=response_a)
    response_b.change(lambda x: update_label(x, response_b), inputs=response_b, outputs=response_b)

    # Flagging
    def flag(prompt, compr_prompt, rate, metrics, res_a_obj, res_b_obj, flag_button, request: gr.Request):
        args = [prompt, compr_prompt, rate, metrics, res_a_obj, res_b_obj]
        flagging_callback.flag(args, flag_option=flag_button[0], username=request.cookies["session"])
        gr.Info("Preference saved. Thank you for your feedback.")
        return [activate_button(False)] * 3

    FLAG_COMPONENTS = [prompt, compressed_prompt, rate, metrics, response_a_obj, response_b_obj]
    flagging_callback.setup(FLAG_COMPONENTS, "flagged")
    response_a.change(activate_button, inputs=response_a, outputs=button_a)
    response_a.change(activate_button, inputs=response_a, outputs=button_ab)
    response_b.change(activate_button, inputs=response_b, outputs=button_b)
    button_a.click(flag, inputs=FLAG_COMPONENTS + [button_a], outputs=[button_a, button_ab, button_b], preprocess=False)
    button_ab.click(
        flag, inputs=FLAG_COMPONENTS + [button_ab], outputs=[button_a, button_ab, button_b], preprocess=False
    )
    button_b.click(flag, inputs=FLAG_COMPONENTS + [button_b], outputs=[button_a, button_ab, button_b], preprocess=False)

    # Examples
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
