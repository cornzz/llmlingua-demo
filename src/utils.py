import json
import os
from random import shuffle
from secrets import compare_digest

import gradio as gr
import pandas as pd
import requests
from fastapi import HTTPException


class DiffSeparator(str):
    def __init__(self, sep):
        self.sep = sep

    def __add__(self, text):
        if text in """!@#$%^&*()_+\-=\[\]{};':"\|,.<>\/?`~""":
            return text
        else:
            return self.sep + text


def get_api_info() -> list[str]:
    endpoint, token = os.getenv("LLM_ENDPOINT"), os.getenv("LLM_TOKEN")
    if not endpoint:
        print("LLM_ENDPOINT environment variable is not set, only compression will be possible...")
        models = []
    else:
        if not token:
            print("LLM_TOKEN environment variable is not set, will use API without token...")
        models = [m.strip() for m in (os.getenv("LLM_LIST") or "").split(",") if m]
        if not models:
            headers = {"Content-Type": "application/json", "Authorization": f"Bearer {token or 'no-key'}"}
            response = requests.get(f"{endpoint}/models", headers=headers)
            if response.status_code != 200:
                print(f"Error while loading models from API: {response.status_code} - {response.text}")
            else:
                models = [model["id"] for model in response.json()["data"]]
    return endpoint, token, models


def create_metrics_df(result: dict = None) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "Original / Compressed": (
                [f'{result["origin_tokens"]} / {result["compressed_tokens"]} (tokens)'] if result else [""]
            ),
            "Ratio": [result["ratio"]] if result else [""],
            "Rate": [result["rate"]] if result else [""],
            "Saving": (
                [f"${(result['origin_tokens'] - result['compressed_tokens']) * 0.03 / 1000:.4f} in GPT-4"]
                if result
                else [""]
            ),
            "Compression": [""],
            "End-to-end Latency": [""],
            "End-to-end Latency Compressed (Speedup)": [""],
        }
    )
    return df


def activate_button(value: str) -> gr.Button:
    return gr.Button(interactive=bool(value))


def handle_tabs(event: gr.SelectData) -> tuple[bool, gr.Textbox, gr.Radio, gr.Row, gr.Row]:
    compress_only = event.value == "Compress only"
    return (
        compress_only,
        gr.Textbox(visible=not compress_only),
        gr.Radio(visible=not compress_only),
        gr.Row(visible=not compress_only),
    )


def update_label(content: str, component: gr.Textbox | gr.HighlightedText) -> gr.Textbox | gr.HighlightedText:
    words = len(content.split())
    new_label = component.label.split(" (")[0] + (f" ({words} words)" if words else "")
    return gr.Textbox(label=new_label) if isinstance(component, gr.Textbox) else gr.HighlightedText(label=new_label)


def shuffle_and_flatten(compressed: dict[str, object], original: dict[str, object], survey_mode: bool) -> iter:
    responses = [compressed, original]
    if survey_mode:
        shuffle(responses)
    return (x for xs in responses for x in xs.values())


def get_message(response: dict) -> str:
    if "choices" in response:
        return response["choices"][0]["message"]["content"]
    elif "error" in response:
        res_text = f'Error calling LLM API: {response["code"]} - {response["error"]}'
        gr.Warning(res_text)
        return res_text


def create_llm_response(response: requests.Response, compressed: bool, start: float, end: float):
    # TODO: improve this mess... (use openai library?)
    if response.status_code == 200:
        response = response.json()
        error = "error" in response and response["error"]
    else:
        error = {"code": response.status_code, "message": response.text}
    obj = {
        "compressed": compressed,
        "call_time": end - start,
        "error": (
            (error["message"]["error"] if type(error["message"]) == dict else error["message"]) if error else False
        ),
        "code": error["code"] if error else 200,
    }
    if not error:
        obj.update(response)
    return {"text": get_message(obj), "obj": obj}


def metrics_to_df(metrics: dict):
    return pd.DataFrame(metrics["data"], columns=metrics["headers"])


def prepare_flagged_data(data: pd.DataFrame):
    if not data.empty:
        data["Response A"] = data["Response A"].apply(json.loads)
        data["Response B"] = data["Response B"].apply(json.loads)
        data.insert(5, "Model", data["Response A"].apply(lambda x: x["model"]))
        # Switch responses so the uncompressed one is always on the left
        data["flag"] = data.apply(
            lambda r: "".join(
                [
                    "✅" if y else "❌"
                    for y in (json.loads(r["flag"])[::-1] if r["Response A"]["compressed"] else json.loads(r["flag"]))
                ]
            ),
            axis=1,
        )
        data["Response A"], data["Response B"] = zip(
            *data.apply(
                lambda x: (
                    map(get_message, [x["Response B"], x["Response A"]])
                    if x["Response A"]["compressed"]
                    else map(get_message, [x["Response A"], x["Response B"]])
                ),
                axis=1,
            )
        )
        data["Metrics"] = data["Metrics"].apply(lambda x: metrics_to_df(json.loads(x)).to_html(index=False))
    data = data.rename(
        columns={
            "Response A": "Uncompressed Response",
            "Response B": "Compressed Response",
            "username": "user",
            "timestamp": "time",
        }
    )
    return data.iloc[::-1].to_html(table_id="table")


def check_password(submitted: str, password: str):
    if not (password and compare_digest(submitted.encode("utf8"), password.encode("utf8"))):
        raise HTTPException(
            status_code=401, detail="Invalid or missing credentials", headers={"WWW-Authenticate": "Basic"}
        )


def stream_file(file_path, chunk_size=4096):
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            yield chunk
