import json
import os
import re
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated

from fastapi.staticfiles import StaticFiles
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
    handle_tabs,
    metrics_to_df,
    prepare_flagged_data,
    shuffle_and_flatten,
    stream_file,
    update_label,
)

start_load = time.time()
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONSENT_POPUP = os.getenv("CONSENT_POPUP", "false")
FLAG_DIRECTORY = os.path.join(BASE_DIR, "../flagged")
FLAG_PASSWORD = os.getenv("FLAG_PASSWORD")
LOG_DIRECTORY = os.path.join(FLAG_DIRECTORY, "logs")
build_logger("monitor", LOG_DIRECTORY, "monitor.log")
print("Loading LLMLingua Demo...")
LLM_ENDPOINT, LLM_TOKEN, LLM_LIST = get_api_info()
APP_PATH = os.getenv("APP_PATH") or ""
MPS_AVAILABLE = torch.backends.mps.is_available()
CUDA_AVAILABLE = torch.cuda.is_available()
GH_LOGO = """<svg viewBox="0 0 98 96" xmlns="http://www.w3.org/2000/svg"><path fill-rule="evenodd" clip-rule="evenodd" d="M48.854 0C21.839 0 0 22 0 49.217c0 21.756 13.993 40.172 33.405 46.69 2.427.49 3.316-1.059 3.316-2.362 0-1.141-.08-5.052-.08-9.127-13.59 2.934-16.42-5.867-16.42-5.867-2.184-5.704-5.42-7.17-5.42-7.17-4.448-3.015.324-3.015.324-3.015 4.934.326 7.523 5.052 7.523 5.052 4.367 7.496 11.404 5.378 14.235 4.074.404-3.178 1.699-5.378 3.074-6.6-10.839-1.141-22.243-5.378-22.243-24.283 0-5.378 1.94-9.778 5.014-13.2-.485-1.222-2.184-6.275.486-13.038 0 0 4.125-1.304 13.426 5.052a46.97 46.97 0 0 1 12.214-1.63c4.125 0 8.33.571 12.213 1.63 9.302-6.356 13.427-5.052 13.427-5.052 2.67 6.763.97 11.816.485 13.038 3.155 3.422 5.015 7.822 5.015 13.2 0 18.905-11.404 23.06-22.324 24.283 1.78 1.548 3.316 4.481 3.316 9.126 0 6.6-.08 11.897-.08 13.526 0 1.304.89 2.853 3.316 2.364 19.412-6.52 33.405-24.935 33.405-46.691C97.707 22 75.788 0 48.854 0z" fill="#24292f"/></svg>"""

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


app.mount("/fonts", StaticFiles(directory="fonts"), name="fonts")


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
    return create_llm_response(response, compressed, start, end=time.time())


def compress_prompt(prompt: str, rate: float, force_tokens: list[str], force_digits: bool):
    force_tokens = ["\n" if x == "\\n" else x for x in force_tokens]
    start = time.time()
    result = llm_lingua.compress_prompt(
        prompt, rate=rate, force_tokens=force_tokens, force_reserve_digit=force_digits, return_word_label=True
    )
    compression_time = time.time() - start

    word_sep, label_sep = "\t\t|\t\t", " "
    diff = [
        (word, (None, "+")[int(label)])
        for line in result["fn_labeled_original_prompt"].split(word_sep)
        for word, label in [line.rsplit(label_sep, 1)]
    ]
    return result["compressed_prompt"], diff, create_metrics_df(result), compression_time


def run_demo(
    question: str,
    prompt: str,
    rate: float,
    target_model: str,
    compress_only: bool,
    force_tokens: list[str],
    force_digits: list[str],
    request: gr.Request,
):
    rate = rate / 100
    print(
        f"RUN DEMO - question: {len(question.split())}, prompt: {len(prompt.split())}, rate: {rate},",
        f"model: {'Compress only' if compress_only else target_model.split('/')[-1]} - from {request.cookies['session']}",
    )
    if compress_only:
        compressed, diff, metrics, compression_time = compress_prompt(prompt, rate, force_tokens, bool(force_digits))
        metrics["Compression"] = [f"{compression_time:.2f}s"]
        return [compressed, diff, metrics] + [None] * 4 + [gr.Button(interactive=False)] * 4 + [[None, None]]

    with ThreadPoolExecutor() as executor:
        get_query = lambda p: f"{question}\n\n{p}" if question else p
        future_original = executor.submit(call_llm_api, get_query(prompt), target_model)
        compressed, diff, metrics, compression_time = compress_prompt(prompt, rate, force_tokens, bool(force_digits))
        res_compressed = call_llm_api(get_query(compressed), target_model, True)
        res_original = future_original.result()

    end_to_end_original = res_original["obj"]["call_time"]
    end_to_end_compressed = res_compressed["obj"]["call_time"] + compression_time
    metrics["Compression"] = [f"{compression_time:.2f}s"]
    metrics["End-to-end Latency"] = [f"{end_to_end_original:.2f}s"]
    metrics["End-to-end Latency Compressed (Speedup)"] = [
        f"{end_to_end_compressed:.2f}s ({end_to_end_original / end_to_end_compressed:.2f}x)"
    ]
    error = res_original["obj"]["error"] or res_compressed["obj"]["error"]
    res_original["obj"], res_compressed["obj"] = json.dumps(res_original["obj"]), json.dumps(res_compressed["obj"])
    return (
        [
            compressed,
            diff,
            metrics,
            *shuffle_and_flatten(res_original, res_compressed),
        ]
        + [gr.Button(interactive=not error)] * 4
        + [[None, None]]
    )


with gr.Blocks(
    title="LLMLingua-2 Demo",
    css=os.path.join(BASE_DIR, "app.css"),
    js=os.path.join(BASE_DIR, "app.js"),
    head=f"""
        <link rel=\"icon\" href=\"favicon.ico\">
        <script>const CONSENT_POPUP = {CONSENT_POPUP}</script>
    """,
    analytics_enabled=False,
    theme=gr.themes.Default(font="Source Sans 3", font_mono="IBM Plex Mono"),
) as demo:
    gr.Markdown(
        f'# Prompt Compression Demo <a class="source" href="https://github.com/cornzz/llmlingua-demo" target="_blank">{GH_LOGO}</a>'
    )
    # Info / Settings
    with gr.Accordion("About this demo (please read):", open=False, elem_classes="accordion"):
        gr.Markdown(
            "Your prompt is sent to a target LLM for completion, both in its uncompressed form and compressed using [LLMLingua-2](https://llmlingua.com/llmlingua2.html). "
            "Evaluate the responses and give feedback for each one by clicking on the respective button below the answer."
        )
        gr.Markdown(
            f"""
                - **The order of the responses (prompt compressed / uncompressed) is randomized**.
                - LLMLingua-2 is a task-agnostic compression model, the value of the question field is not considered in the compression process. Compression is performed {'on a CPU. Using a GPU would be faster.' if not (MPS_AVAILABLE or CUDA_AVAILABLE) else f'on a GPU {"using MPS." if MPS_AVAILABLE else f"({torch.cuda.get_device_name()})."}'}
                - The example prompts were taken from the [MeetingBank-QA-Summary](https://huggingface.co/datasets/microsoft/MeetingBank-QA-Summary) dataset. Click on a question to autofill the question field.
                - Token counts are calculated using the [cl100k_base tokenizer](https://platform.openai.com/tokenizer) (GPT-3.5/-4), actual counts may vary for different target models. The saving metric is based on an API pricing of $0.03 / 1000 tokens.
                - End-to-end latency: latency from submission to full response, including compression. While shown for reference, this metric alone is not an effective measure of compression efficacy.
            """,
            elem_id="notes",
        )
        with gr.Column(variant="compact", elem_id="settings"):
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

    # Examples
    with gr.Accordion("Example prompts:", open=False, elem_classes="accordion"):
        examples = gr.Dataset(
            samples=[[example["original_prompt"]] for example in example_dataset],
            components=[gr.Textbox(visible=False)],
            samples_per_page=3,
            type="index",
            elem_id="examples",
        )
        examples_back = gr.Button("← Choose different example", elem_classes="link-button", visible=False)
        qa_pairs = gr.Dataframe(
            label="QA pairs related to the selected example prompt. Click on a question to autofill the question field.",
            headers=["Question", "Answer"],
            elem_classes="qa-pairs dataframe",
            visible=False,
        )

    # Inputs
    tab_prompt, tab_compress = gr.Tab("Prompt target LLM", id=0), gr.Tab("Compress only", id=1)
    question = gr.Textbox(
        label="Question",
        info="(will not be compressed)",
        lines=1,
        placeholder=example_dataset[1]["QA_pairs"][6][0],
        elem_classes="question-target",
    )
    prompt = gr.Textbox(
        label="Prompt / Context",
        lines=8,
        max_lines=8,
        autoscroll=False,
        placeholder=example_dataset[1]["original_prompt"],
        elem_classes="word-count",
    )
    rate = gr.Slider(10, 100, 50, step=1, label="Rate", info="(compression target)", elem_classes="rate")
    target_model = gr.Radio(label="Target LLM", choices=LLM_LIST, value=LLM_LIST[0])
    with gr.Row():
        clear = gr.Button("Clear", elem_classes="clear")
        submit = gr.Button("Submit", variant="primary", interactive=False)

    # Outputs
    with gr.Column(variant="panel", elem_classes="outputs"):
        gr.Markdown('<h2 style="text-align: center">Results</h2>')
        metrics = gr.Dataframe(
            label="Metrics",
            headers=[*create_metrics_df().columns],
            row_count=1,
            height=75,
            column_widths=["17.8%", "6.8%", "6.1%", "12.3%", "11%", "15.7%", "30.3%"],
            show_label=False,
            interactive=False,
            elem_classes="metrics dataframe",
        )
        compressedDiff = gr.HighlightedText(
            label="Compressed Prompt",
            show_inline_category=False,
            combine_adjacent=True,
            adjacent_separator=DiffSeparator(" "),
            color_map={"+": "green"},
            elem_id="compressed-diff",
        )
        compressed = gr.Textbox(label="Compressed Prompt", visible=False)
        with gr.Row(elem_classes="responses") as responses:
            response_a = gr.Textbox(label="LLM Response A", lines=10, max_lines=10, autoscroll=False, interactive=False)
            response_a_obj = gr.Textbox(label="Response A", visible=False)
            response_b = gr.Textbox(label="LLM Response B", lines=10, max_lines=10, autoscroll=False, interactive=False)
            response_b_obj = gr.Textbox(label="Response B", visible=False)
        with gr.Column() as flag_buttons:
            with gr.Row():
                a_yes = gr.Button("✅", interactive=False)
                a_no = gr.Button("❌", interactive=False)
                b_yes = gr.Button("✅", interactive=False)
                b_no = gr.Button("❌", interactive=False)
                FLAG_BUTTONS = [a_yes, a_no, b_yes, b_no]
            gr.Markdown(
                '<div class="button-hint">✅ = answered your question / solved your problem'
                "&nbsp;&nbsp;&nbsp; ❌ = did not answer your question / solve your problem.</div>"
            )

    # States
    compress_only, flags = gr.State(False), gr.State([None, None])

    # Event handlers
    for tab in [tab_prompt, tab_compress]:
        tab.select(handle_tabs, outputs=[compress_only, question, target_model, responses, flag_buttons], js="openDiff")
    prompt.change(activate_button, inputs=[prompt], outputs=[submit])
    submit.click(
        run_demo,
        inputs=[question, prompt, rate, target_model, compress_only, force_tokens, force_digits],
        outputs=[
            compressed,
            compressedDiff,
            metrics,
            response_a,
            response_a_obj,
            response_b,
            response_b_obj,
            *FLAG_BUTTONS,
            flags,
        ],
    )
    clear.click(
        lambda: [None] * 8
        + [50, create_metrics_df(), gr.DataFrame(visible=False)]
        + [gr.Button(interactive=False)] * 4
        + [[None, None]],
        outputs=[
            question,
            prompt,
            compressed,
            compressedDiff,
            response_a_obj,
            response_a,
            response_b_obj,
            response_b,
            rate,
            metrics,
            qa_pairs,
            *FLAG_BUTTONS,
            flags,
        ],
    )
    compressed.change(lambda x: update_label(x, compressedDiff), inputs=[compressed], outputs=[compressedDiff])
    response_a.change(lambda x: update_label(x, response_a), inputs=[response_a], outputs=[response_a])
    response_b.change(lambda x: update_label(x, response_b), inputs=[response_b], outputs=[response_b])
    examples.select(
        lambda idx: (
            None,
            example_dataset[idx]["original_prompt"],
            gr.Dataset(visible=False),
            gr.Button(visible=True),
            (
                gr.DataFrame(example_dataset[idx]["QA_pairs"], visible=True)
                if "QA_pairs" in example_dataset[idx]
                else gr.DataFrame(visible=False)
            ),
        ),
        inputs=[examples],
        outputs=[question, prompt, examples, examples_back, qa_pairs],
    )
    examples_back.click(
        lambda: (gr.Dataset(visible=True), gr.Button(visible=False), gr.DataFrame(visible=False)),
        outputs=[examples, examples_back, qa_pairs],
    )

    # Flagging
    def handle_flag_selection(question, prompt, compressed, rate, metrics, res_a, res_b, flags, request: gr.Request):
        if None in flags:
            return
        metrics = gr.DataFrame().postprocess(metrics).__dict__
        model = re.search(r'model": "(?:[^/]*\/)?(.*?)", "object', res_a)
        model = model.group(1) if model else ""
        print(f"FLAG - model: {model}, flags: {flags} - from {request.cookies['session']}")
        args = [question, prompt, compressed, rate / 100, metrics, res_a, res_b]
        flagging_callback.flag(args, flag_option=json.dumps(flags), username=request.cookies["session"])
        gr.Info("Preference saved. Thank you for your feedback.")

    def flag(response: str, value: bool, fs: list[bool]):
        fs[response == "B"] = value
        return [gr.Button(interactive=False)] * 2 + [fs]

    FLAG_COMPONENTS = [question, prompt, compressed, rate, metrics, response_a_obj, response_b_obj]
    flagging_callback.setup(FLAG_COMPONENTS, FLAG_DIRECTORY)
    a_yes.click(lambda fs: flag("A", True, fs), inputs=[flags], outputs=FLAG_BUTTONS[:2] + [flags])
    a_no.click(lambda fs: flag("A", False, fs), inputs=[flags], outputs=FLAG_BUTTONS[:2] + [flags])
    b_yes.click(lambda fs: flag("B", True, fs), inputs=[flags], outputs=FLAG_BUTTONS[2:] + [flags])
    b_no.click(lambda fs: flag("B", False, fs), inputs=[flags], outputs=FLAG_BUTTONS[2:] + [flags])
    flags.change(handle_flag_selection, inputs=FLAG_COMPONENTS + [flags])


app = gr.mount_gradio_app(app, demo, path="/", root_path=APP_PATH)
print(f"Ready! Loaded in {time.time() - start_load:.2f}s")
