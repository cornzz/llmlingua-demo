import json
import os
import re
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated

import gradio as gr
import pandas as pd
import requests
import torch
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from llmlingua import PromptCompressor

from .logger import build_logger
from .utils import (
    DiffSeparator,
    activate_button,
    check_password,
    create_llm_response,
    create_metrics_df,
    get_api_info,
    handle_model_change,
    handle_ui_settings,
    metrics_to_df,
    prepare_flagged_data,
    shuffle_and_flatten,
    stream_file,
    update_label,
)

start_load = time.time()
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FLAG_DIRECTORY = os.path.join(BASE_DIR, "../flagged")
FLAG_PASSWORD = os.getenv("FLAG_PASSWORD")
LOG_DIRECTORY = os.path.join(FLAG_DIRECTORY, "logs")
build_logger("monitor", LOG_DIRECTORY, "monitor.log")
print("Loading LLMLingua Demo...")
LLM_ENDPOINT, LLM_TOKEN, LLM_LIST = get_api_info()
APP_PATH = os.getenv("APP_PATH") or ""
MPS_AVAILABLE = torch.backends.mps.is_available()
CUDA_AVAILABLE = torch.cuda.is_available()

app = FastAPI(openapi_url="", root_path=APP_PATH)
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
        data = prepare_flagged_data(pd.read_csv(FLAG_DIRECTORY + "/log.csv", na_filter=False))
        with open(os.path.join(BASE_DIR, "flagged.html")) as f:
            return f.read().replace("{{ data }}", data)


@app.get("/flagged/download")
def download_flagged(credentials: Annotated[HTTPBasicCredentials, Depends(HTTPBasic())]):
    check_password(credentials.password, FLAG_PASSWORD)
    if os.path.exists(FLAG_DIRECTORY + "/log.csv"):
        return FileResponse(path=FLAG_DIRECTORY + "/log.csv", filename="flagged.csv", media_type="text/csv")


@app.delete("/flagged/delete/{index}")
def delete_flagged(index: int, credentials: Annotated[HTTPBasicCredentials, Depends(HTTPBasic())]):
    check_password(credentials.password, FLAG_PASSWORD)
    if os.path.exists(FLAG_DIRECTORY + "/log.csv"):
        data = pd.read_csv(FLAG_DIRECTORY + "/log.csv", na_filter=False)
        if index < 0 or index >= len(data):
            raise HTTPException(status_code=404, detail="Index out of range")
        data.drop(index, inplace=True)
        data.to_csv(FLAG_DIRECTORY + "/log.csv", index=False)


@app.get("/flagged/{index}")
def get_flagged(index: int, credentials: Annotated[HTTPBasicCredentials, Depends(HTTPBasic())]):
    check_password(credentials.password, FLAG_PASSWORD)
    if os.path.exists(FLAG_DIRECTORY + "/log.csv"):
        try:
            data = pd.read_csv(
                FLAG_DIRECTORY + "/log.csv", skiprows=lambda x: x != 0 and x - 1 != index, na_filter=False
            )
            data["Metrics"] = data["Metrics"].apply(lambda x: metrics_to_df(json.loads(x)).to_dict(orient="records")[0])
            data["Response A"] = data["Response A"].apply(json.loads)
            data["Response B"] = data["Response B"].apply(json.loads)
            return data.to_dict(orient="records")[0]
        except Exception:
            raise HTTPException(status_code=404, detail="Index out of range")


@app.get("/logs", response_class=HTMLResponse)
def get_logs(credentials: Annotated[HTTPBasicCredentials, Depends(HTTPBasic())]):
    check_password(credentials.password, FLAG_PASSWORD)
    if os.path.exists(LOG_DIRECTORY):
        logs = "<br>".join([f"<a href='logs/{log_file}'>{log_file}</a>" for log_file in os.listdir(LOG_DIRECTORY)])
        return f"{logs}<br>-----<br><a href='logs/download'>Download all</a>"


@app.get("/logs/download")
def download_logs(credentials: Annotated[HTTPBasicCredentials, Depends(HTTPBasic())]):
    check_password(credentials.password, FLAG_PASSWORD)
    if os.path.exists(LOG_DIRECTORY):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_file:
            zip_path = tmp_file.name
            shutil.make_archive(base_name=zip_path.replace(".zip", ""), format="zip", root_dir=LOG_DIRECTORY)
        return FileResponse(path=zip_path, media_type="application/zip", filename="logs.zip")


@app.get("/logs/{log_name}")
def get_logs(log_name: str, credentials: Annotated[HTTPBasicCredentials, Depends(HTTPBasic())]):
    check_password(credentials.password, FLAG_PASSWORD)
    log_path = os.path.join(LOG_DIRECTORY, log_name.replace("/", ""))
    if os.path.exists(log_path):
        return StreamingResponse(stream_file(log_path), media_type="text/plain")


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
    response = requests.post(f"{LLM_ENDPOINT}/chat/completions", headers=headers, data=data)
    if response.status_code != 200:
        gr.Warning(f"Error calling LLM API: {response.status_code} - {response.text}")
    return create_llm_response(response, compressed, response.status_code != 200, start, end=time.time())


def compress_prompt(prompt: str, rate: float, force_tokens: list[str], force_digits: bool):
    force_tokens = ["\n" if x == "\\n" else x for x in force_tokens]
    start = time.time()
    result = llm_lingua.compress_prompt(
        prompt, rate=rate, force_tokens=force_tokens, force_reserve_digit=force_digits, return_word_label=True
    )
    compression_time = time.time() - start

    word_sep, label_sep = "\t\t|\t\t", " "
    diff = []
    for line in result["fn_labeled_original_prompt"].split(word_sep):
        word, label = line.split(label_sep)
        diff.append((word, "+") if label == "1" else (word, None))
    return result["compressed_prompt"], diff, create_metrics_df(result), compression_time


def run_demo(
    prompt: str,
    context: str,
    rate: float,
    target_model: str,
    force_tokens: list[str],
    force_digits: list[str],
    request: gr.Request,
):
    print(
        f"RUN DEMO - prompt: {len(prompt.split())}, context: {len(context.split())}, rate: {rate},",
        f"model: {target_model.split('/')[-1]} - from {request.cookies['session']}",
    )
    if target_model == "Compress only":
        compressed, diff, metrics, compression_time = compress_prompt(context, rate, force_tokens, bool(force_digits))
        metrics["Compression"] = [f"{compression_time:.2f}s"]
        return compressed, diff, metrics, None, None, None, None

    with ThreadPoolExecutor() as executor:
        future_original = executor.submit(
            call_llm_api, "\n\n".join([prompt, context]) if prompt else context, target_model
        )
        compressed, diff, metrics, compression_time = compress_prompt(context, rate, force_tokens, bool(force_digits))
        res_compressed = call_llm_api("\n\n".join([prompt, compressed]) if prompt else compressed, target_model, True)
        res_original = future_original.result()

    end_to_end_original = res_original["obj"]["call_time"]
    end_to_end_compressed = res_compressed["obj"]["call_time"] + compression_time
    metrics["Compression"] = [f"{compression_time:.2f}s"]
    metrics["End-to-end Latency"] = [f"{end_to_end_original:.2f}s"]
    metrics["End-to-end Latency Compressed (Speedup)"] = [
        f"{end_to_end_compressed:.2f}s ({end_to_end_original / end_to_end_compressed:.2f}x)"
    ]
    res_original["obj"], res_compressed["obj"] = json.dumps(res_original["obj"]), json.dumps(res_compressed["obj"])
    return compressed, diff, metrics, *shuffle_and_flatten(res_original, res_compressed)


with gr.Blocks(
    title="LLMLingua Demo", css=os.path.join(BASE_DIR, "app.css"), js=os.path.join(BASE_DIR, "app.js")
) as demo:
    gr.Markdown("# Prompt Compression A/B Test")
    with gr.Accordion("About this demo (please read):", open=False, elem_classes="accordion"):
        gr.Markdown(
            f"""
            Your prompt is sent to a target LLM for completion, both in its uncompressed form and compressed using [LLMLingua-2](https://llmlingua.com/llmlingua2.html). Evaluate the responses and select the better one.
            Notes:
            - The order of the responses (compressed / uncompressed prompt) is randomized.
            - Compression time is included in the compressed end-to-end latency. {'Compression is done on a CPU. Using a GPU would be faster.' if not (MPS_AVAILABLE or CUDA_AVAILABLE) else 'Compression is done on a GPU using MPS.' if MPS_AVAILABLE else f'Compression is done on a GPU ({torch.cuda.get_device_name()}).'}
            - The provided example prompts were taken from the [LLMLingua-2 Demo](https://huggingface.co/spaces/microsoft/llmlingua-2). Some include corresponding QA pairs generated by GPT-4. Click on a question to autofill the separate, uncompressed question field.
            - This demo focuses on evaluating the quality of responses to compressed prompts. Uncompressed and compressed prompts are processed simultaneously. Thus, and due to other variables, the end-to-end latencies may not be very meaningful.
            - Token counts are calculated based on the `cl100k_base` [tokenizer](https://platform.openai.com/tokenizer) (GPT-3.5/-4) and may vary for different target models. Saving metric is based on an API pricing of $0.03 / 1000 tokens.
            - LLMLingua-2 is a task-agnostic compression model, so the value of the question field is not considered in the compression process.
        """
        )
        with gr.Row(variant="compact"):
            with gr.Column():
                gr.Markdown("UI Settings")
                ui_settings = gr.CheckboxGroup(
                    ["Show Metrics", "Show Separate Context Field", "Show Compressed Prompt"],
                    container=False,
                    value=["Show Metrics", "Show Separate Context Field"],
                    elem_classes="ui-settings",
                )
            with gr.Column():
                gr.Markdown("Tokens to Preserve")
                with gr.Row():
                    force_tokens = gr.Dropdown(
                        show_label=False,
                        container=False,
                        choices=["\\n", ".", "!", "?", ","],
                        value=["\\n"],
                        multiselect=True,
                        allow_custom_value=True,
                        scale=3,
                        elem_classes="force-tokens",
                    )
                    force_digits = gr.CheckboxGroup(
                        ["Preserve Digits"], show_label=False, container=False, value=[], elem_classes="digits-checkbox"
                    )

    # Inputs
    prompt = gr.Textbox(label="Question", lines=1, max_lines=1, elem_classes="question-target")
    context = gr.Textbox(label="Context", lines=8, max_lines=8, autoscroll=False, elem_classes="word-count")
    rate = gr.Slider(0.1, 1, 0.5, step=0.05, label="Rate")
    target_model = gr.Radio(label="Target LLM", choices=LLM_LIST, value=LLM_LIST[0])
    with gr.Row():
        clear = gr.Button("Clear", elem_classes="clear")
        submit = gr.Button("Submit", variant="primary", interactive=False)

    # Outputs
    metrics = gr.Dataframe(
        label="Metrics",
        headers=[*create_metrics_df().columns],
        row_count=1,
        height=90,
        show_label=False,
        interactive=False,
    )
    compressed = gr.Textbox(label="Compressed Prompt", lines=2, max_lines=2, visible=False, interactive=False)
    compressedDiff = gr.HighlightedText(
        label="Compressed Prompt",
        visible=False,
        show_inline_category=False,
        combine_adjacent=True,
        adjacent_separator=DiffSeparator(" "),
        color_map={"+": "green"},
        elem_id="compressed-diff",
        elem_classes="no-content",
    )
    with gr.Column(variant="panel") as responses:
        # TODO: stream response?
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

    # Event handlers
    prompt.change(activate_button, inputs=[prompt, context], outputs=submit)
    context.change(activate_button, inputs=[prompt, context], outputs=submit)
    submit.click(
        run_demo,
        inputs=[prompt, context, rate, target_model, force_tokens, force_digits],
        outputs=[compressed, compressedDiff, metrics, response_a, response_a_obj, response_b, response_b_obj],
    )
    clear.click(
        lambda: [None] * 8 + [0.5, create_metrics_df(), gr.DataFrame(visible=False)],
        outputs=[
            prompt,
            context,
            compressed,
            compressedDiff,
            response_a_obj,
            response_a,
            response_b_obj,
            response_b,
            rate,
            metrics,
            qa_pairs,
        ],
    )
    ui_settings.change(handle_ui_settings, inputs=ui_settings, outputs=[prompt, context, compressedDiff, metrics])
    target_model.change(handle_model_change, inputs=[target_model, ui_settings], outputs=[compressedDiff, responses])
    compressed.change(lambda x: update_label(x, compressedDiff), inputs=compressed, outputs=compressedDiff)
    response_a.change(lambda x: update_label(x, response_a), inputs=response_a, outputs=response_a)
    response_b.change(lambda x: update_label(x, response_b), inputs=response_b, outputs=response_b)
    response_a.change(activate_button, inputs=response_a, outputs=flag_a)
    response_a.change(activate_button, inputs=response_a, outputs=flag_n)
    response_b.change(activate_button, inputs=response_b, outputs=flag_b)
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
        model = re.search(r'model": "(?:[^/]*\/)?(.*?)", "object', res_a_obj)
        print(
            f"FLAG - model: {model.group(1) if model else ''}, flag: {flag_button[0]} - from {request.cookies['session']}"
        )
        args = [prompt, context, compr_prompt, rate, metrics, res_a_obj, res_b_obj]
        flagging_callback.flag(args, flag_option=flag_button[0], username=request.cookies["session"])
        gr.Info("Preference saved. Thank you for your feedback.")
        return [activate_button(False)] * 3

    FLAG_COMPONENTS = [prompt, context, compressed, rate, metrics, response_a_obj, response_b_obj]
    flagging_callback.setup(FLAG_COMPONENTS, FLAG_DIRECTORY)
    flag_a.click(flag, inputs=FLAG_COMPONENTS + [flag_a], outputs=[flag_a, flag_n, flag_b], preprocess=False)
    flag_n.click(flag, inputs=FLAG_COMPONENTS + [flag_n], outputs=[flag_a, flag_n, flag_b], preprocess=False)
    flag_b.click(flag, inputs=FLAG_COMPONENTS + [flag_b], outputs=[flag_a, flag_n, flag_b], preprocess=False)


app = gr.mount_gradio_app(app, demo, path="/", root_path=APP_PATH)
print(f"Ready! Loaded in {time.time() - start_load:.2f}s")
