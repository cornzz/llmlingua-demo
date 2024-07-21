from llmlingua import PromptCompressor
import gradio as gr
import torch
import pandas as pd
import json
import requests
import os
from dotenv import load_dotenv

load_dotenv()

LLM_ENDPOINT = os.getenv("LLM_ENDPOINT")
LLM_MODELS = ["meta-llama/Meta-Llama-3-70B-Instruct", "mistral-7b-q4", "CohereForAI/c4ai-command-r-plus"]

llm_lingua = PromptCompressor(
    model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
    use_llmlingua2=True,
    device_map="mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu",
)


def create_metrics_df(result=None):
    return pd.DataFrame(
        {
            "Original / Compressed Tokens": (
                [f'{result["origin_tokens"]} / {result["compressed_tokens"]}'] if result else [""]
            ),
            "Ratio": [result["ratio"]] if result else [""],
            "Rate": [result["rate"]] if result else [""],
            "Saving": [result["saving"]] if result else [""],
        }
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
        return response.json()
    else:
        return response.text


def compress_prompt(prompt: str, rate: float):
    result = llm_lingua.compress_prompt(prompt, rate=rate)

    return (
        result["compressed_prompt"],
        create_metrics_df(result),
    )


def run(prompt: str, rate: float, target_model: str):
    compressed_prompt, metrics = compress_prompt(prompt, rate)
    response_compressed = call_llm_api(compressed_prompt, target_model)
    response_original = call_llm_api(prompt, target_model)

    return (
        compressed_prompt,
        metrics,
        response_compressed,
        response_compressed["choices"][0]["message"]["content"],
        response_original,
        response_original["choices"][0]["message"]["content"],
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
    target_model = gr.Dropdown(LLM_MODELS, value=LLM_MODELS[0], label="Target LLM Model")
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
    with gr.Row():
        with gr.Column():
            response_a_full = gr.Textbox(label="Response A", visible=False)
            response_a = gr.Textbox(label="LLM Response A", lines=10, max_lines=10, interactive=False)
            button_a = gr.Button("A is better", interactive=False)
        with gr.Column():
            response_b_full = gr.Textbox(label="Response B", visible=False)
            response_b = gr.Textbox(label="LLM Response B", lines=10, max_lines=10, interactive=False)
            button_b = gr.Button("B is better", interactive=False)

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

    prompt.change(activate_button, inputs=prompt, outputs=submit)
    submit.click(
        run,
        inputs=[prompt, rate, target_model],
        outputs=[compressed_prompt, metrics, response_a_full, response_a, response_b_full, response_b],
    )
    clear.click(
        lambda: [create_metrics_df()] + [None] * 6 + [0.5],
        outputs=[
            metrics,
            prompt,
            compressed_prompt,
            response_a_full,
            response_a,
            response_b_full,
            response_b,
            rate,
        ],
    )

    def flag(flag_value: str, *args):
        flagging_callback.flag(*args, flag_value)
        return [activate_button(False)] * 2

    def activate_button(response):
        return gr.Button(interactive=bool(response))

    FLAG_COMPONENTS = [prompt, compressed_prompt, rate, metrics, response_a_full, response_b_full]
    response_a.change(activate_button, inputs=response_a, outputs=button_a)
    response_b.change(activate_button, inputs=response_b, outputs=button_b)
    flagging_callback.setup(FLAG_COMPONENTS, "flagged")
    button_a.click(
        lambda *args: flag("A", args), inputs=FLAG_COMPONENTS, outputs=[button_a, button_b], preprocess=False
    )
    button_b.click(
        lambda *args: flag("B", args), inputs=FLAG_COMPONENTS, outputs=[button_a, button_b], preprocess=False
    )


demo.launch()
