import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from random import shuffle
from typing import Annotated

import gradio as gr
import pandas as pd
import requests
import torch
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from llmlingua import PromptCompressor

from .utils import (
    activate_button,
    check_password,
    create_llm_response,
    create_metrics_df,
    flatten,
    handle_ui_options,
    metrics_to_df,
    prepare_flagged_data,
    update_label,
)

load_dotenv()

LLM_ENDPOINT = os.getenv("LLM_ENDPOINT")
LLM_TOKEN = os.getenv("LLM_TOKEN")
LLM_MODELS = ["meta-llama/Meta-Llama-3.1-70B-Instruct", "mistral-7b-q4", "CohereForAI/c4ai-command-r-plus"]
MPS_AVAILABLE = torch.backends.mps.is_available()
CUDA_AVAILABLE = torch.cuda.is_available()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FLAG_DIRECTORY = os.path.join(BASE_DIR, "../flagged")
FLAG_PASSWORD = os.getenv("FLAG_PASSWORD")
with open(os.path.join(BASE_DIR, "app.js")) as f:
    JS = f.read()
with open(os.path.join(BASE_DIR, "app.css")) as f:
    CSS = f.read()

if not LLM_ENDPOINT:
    print("LLM_ENDPOINT environment variable is not set. Exiting...")
    sys.exit(1)

app = FastAPI()
llm_lingua = PromptCompressor(
    model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
    use_llmlingua2=True,
    device_map="mps" if MPS_AVAILABLE else "cuda" if CUDA_AVAILABLE else "cpu",
)
flagging_callback = gr.CSVLogger()
with open(os.path.join(BASE_DIR, "../data/examples.json")) as f:
    example_dataset = json.load(f)


@app.get("/flagged", response_class=HTMLResponse)
def get_flagged(credentials: Annotated[HTTPBasicCredentials, Depends(HTTPBasic())]):
    check_password(credentials.password, FLAG_PASSWORD)
    if os.path.exists(FLAG_DIRECTORY + "/log.csv"):
        data = pd.read_csv(FLAG_DIRECTORY + "/log.csv")
        with open(os.path.join(BASE_DIR, "flagged.html")) as f:
            return f.read().replace("{{ data }}", prepare_flagged_data(data))


@app.get("/flagged/{index}")
def get_flagged(index: int, credentials: Annotated[HTTPBasicCredentials, Depends(HTTPBasic())]):
    check_password(credentials.password, FLAG_PASSWORD)
    if os.path.exists(FLAG_DIRECTORY + "/log.csv"):
        try:
            data = pd.read_csv(FLAG_DIRECTORY + "/log.csv", skiprows=lambda x: x != 0 and x - 1 != index)
            data["Metrics"] = data["Metrics"].apply(lambda x: metrics_to_df(json.loads(x)).to_dict(orient="records")[0])
            data["Response A"] = data["Response A"].apply(json.loads)
            data["Response B"] = data["Response B"].apply(json.loads)
            return data.to_dict(orient="records")[0]
        except pd.errors.EmptyDataError:
            raise HTTPException(status_code=404, detail="Index out of range")


def call_llm_api(prompt: str, model: str, compressed: bool = False):
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {LLM_TOKEN or 'no-key'}"}
    data = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1000,
        }
    )
    start = time.time()
    response = requests.post(LLM_ENDPOINT, headers=headers, data=data)
    if response.status_code != 200:
        gr.Warning(f"Error calling LLM API: {response.status_code} - {response.text}")
    return create_llm_response(response, compressed, response.status_code != 200, start, end=time.time())


def compress_prompt(prompt: str, rate: float):
    start = time.time()
    result = llm_lingua.compress_prompt(prompt, rate=rate)
    compression_time = time.time() - start

    return result["compressed_prompt"], create_metrics_df(result), compression_time


def run_demo(prompt: str, context: str, rate: float, target_model: str):
    # TODO: allow selecting parallel / sequential processing
    with ThreadPoolExecutor() as executor:
        start = time.time()
        future_original = executor.submit(
            call_llm_api, "\n\n".join([prompt, context]) if prompt else context, target_model
        )
        compressed, metrics, compression_time = compress_prompt(context, rate)
        responses = [
            call_llm_api("\n\n".join([prompt, compressed]) if prompt else compressed, target_model, True),
            future_original.result(),
        ]
        print(f"Processing time: {time.time() - start:.2f}s")

    end_to_end_original = responses[1][1]["call_time"]
    end_to_end_compressed = responses[0][1]["call_time"] + compression_time
    metrics["Compression"] = [f"{compression_time:.2f}s"]
    metrics["End-to-end Latency"] = [f"{end_to_end_original:.2f}s"]
    metrics["End-to-end Latency Compressed (Speedup)"] = [
        f"{end_to_end_compressed:.2f}s ({end_to_end_original / end_to_end_compressed:.2f}x)"
    ]
    shuffle(responses)
    responses[0][1], responses[1][1] = json.dumps(responses[0][1]), json.dumps(responses[1][1])
    return compressed, metrics, *flatten(responses)


with gr.Blocks(title="LLMLingua Demo", css=CSS, js=JS) as demo:
    gr.Markdown("# Prompt Compression A/B Test")
    with gr.Accordion("About this demo (please read):", open=False, elem_classes="accordion"):
        gr.Markdown(
            # TODO: include note that token count may be inaccurate due to different tokenization methods
            # TODO: include info which GPU used
            f"""
            Your prompt is sent to a target LLM model for completion, once uncompressed and once compressed using LLMLingua-2. Compare the responses and select the better one.
            Notes:
            - The order of the responses (compressed / uncompressed prompt) is randomized.
            - Compression time is included in the compressed end-to-end latency.
            {'- Compression is done on a CPU. Using a GPU would be faster.' if not (MPS_AVAILABLE or CUDA_AVAILABLE) else ""}
            - The provided example prompts were taken from the [LLMLingua-2 Demo](https://huggingface.co/spaces/microsoft/llmlingua-2). Some include corresponding QA pairs generated by GPT-4. Click on a question to autofill a separate, uncompressed prompt field.
            - This demo primarily focuses on evaluating the quality of responses to compressed prompts. Uncompressed and compressed prompts are processed simultaneously; thus, and due to other variables, the displayed end-to-end latencies may not be very meaningful.
            - Submitted data is logged if you flag a response (i.e. click on one of the \"x is better\" buttons).
        """
        )
        ui_settings = gr.CheckboxGroup(
            ["Show Separate Context Field", "Show Compressed Prompt", "Show Metrics"],
            label="UI Settings",
            value=["Show Separate Context Field", "Show Metrics"],
            elem_classes="ui-settings",
        )
    prompt = gr.Textbox(
        label="Question (optional, will not be compressed)", lines=1, max_lines=1, elem_classes="question-target"
    )
    context = gr.Textbox(label="Context", lines=8, max_lines=8, autoscroll=False, elem_classes="word-count")
    rate = gr.Slider(0.1, 1, 0.5, step=0.05, label="Rate")
    target_model = gr.Radio(label="Target LLM Model", choices=LLM_MODELS, value=LLM_MODELS[0])
    with gr.Row():
        clear = gr.Button("Clear", elem_classes="clear")
        submit = gr.Button("Submit", variant="primary", interactive=False)

    compressed = gr.Textbox(label="Compressed Prompt", visible=False, interactive=False)
    metrics = gr.Dataframe(
        label="Metrics",
        headers=[*create_metrics_df().columns],
        row_count=1,
        height=90,
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
            flag_a = gr.Button("A is better", interactive=False)
            flag_n = gr.Button("Neither is better", interactive=False)
            flag_b = gr.Button("B is better", interactive=False)

    # Examples
    gr.Markdown("## Examples (click to select)")
    qa_pairs = gr.Dataframe(
        label="GPT-4 generated QA pairs related to the selected example prompt:",
        headers=["Question", "Answer"],
        elem_classes="qa-pairs",
        visible=False,
    )
    examples = gr.Dataset(
        samples=[[example["original_prompt"]] for example in example_dataset],
        components=[gr.Textbox(visible=False)],
        samples_per_page=5,
        type="index",
    )

    # Event handling
    prompt.change(activate_button, inputs=[prompt, context], outputs=submit)
    context.change(activate_button, inputs=[prompt, context], outputs=submit)
    submit.click(
        run_demo,
        inputs=[prompt, context, rate, target_model],
        outputs=[compressed, metrics, response_a, response_a_obj, response_b, response_b_obj],
    )
    clear.click(
        lambda: [None] * 7 + [0.5, create_metrics_df(), gr.DataFrame(visible=False)],
        outputs=[
            prompt,
            context,
            compressed,
            response_a_obj,
            response_a,
            response_b_obj,
            response_b,
            rate,
            metrics,
            qa_pairs,
        ],
    )
    ui_settings.change(handle_ui_options, inputs=ui_settings, outputs=[prompt, context, compressed, metrics])
    compressed.change(lambda x: update_label(x, compressed), inputs=compressed, outputs=compressed)
    response_a.change(lambda x: update_label(x, response_a), inputs=response_a, outputs=response_a)
    response_b.change(lambda x: update_label(x, response_b), inputs=response_b, outputs=response_b)
    examples.select(
        lambda idx: (
            None,
            example_dataset[idx]["original_prompt"],
            (
                gr.DataFrame(example_dataset[idx]["QA_pairs"], visible=True)
                if "QA_pairs" in example_dataset[idx]
                else gr.DataFrame(visible=False)
            ),
        ),
        inputs=examples,
        outputs=[prompt, context, qa_pairs],
    )

    # Flagging
    def flag(prompt, context, compr_prompt, rate, metrics, res_a_obj, res_b_obj, flag_button, request: gr.Request):
        args = [prompt, context, compr_prompt, rate, metrics, res_a_obj, res_b_obj]
        flagging_callback.flag(args, flag_option=flag_button[0], username=request.cookies["session"])
        gr.Info("Preference saved. Thank you for your feedback.")
        return [activate_button(False)] * 3

    FLAG_COMPONENTS = [prompt, context, compressed, rate, metrics, response_a_obj, response_b_obj]
    flagging_callback.setup(FLAG_COMPONENTS, FLAG_DIRECTORY)
    response_a.change(activate_button, inputs=response_a, outputs=flag_a)
    response_a.change(activate_button, inputs=response_a, outputs=flag_n)
    response_b.change(activate_button, inputs=response_b, outputs=flag_b)
    flag_a.click(flag, inputs=FLAG_COMPONENTS + [flag_a], outputs=[flag_a, flag_n, flag_b], preprocess=False)
    flag_n.click(flag, inputs=FLAG_COMPONENTS + [flag_n], outputs=[flag_a, flag_n, flag_b], preprocess=False)
    flag_b.click(flag, inputs=FLAG_COMPONENTS + [flag_b], outputs=[flag_a, flag_n, flag_b], preprocess=False)


app = gr.mount_gradio_app(app, demo, path="/")