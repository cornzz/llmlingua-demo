import json
from random import shuffle
from secrets import compare_digest

import gradio as gr
import pandas as pd
from fastapi import HTTPException
from requests import Response


class DiffSeparator(str):
    def __init__(self, sep):
        self.sep = sep

    def __add__(self, text):
        if text in """!@#$%^&*()_+\-=\[\]{};':"\|,.<>\/?`~""":
            return text
        else:
            return self.sep + text


def create_metrics_df(result: dict = None):
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


def activate_button(*values):
    return gr.Button(interactive=any(bool(value) for value in values))


def handle_ui_settings(value: list[str]):
    show_question = "Show Separate Context Field" in value
    return (
        gr.Textbox(visible=True) if show_question else gr.Textbox(visible=False, value=None),
        gr.Textbox(label="Context" if show_question else "Prompt"),
        gr.HighlightedText(visible="Show Compressed Prompt" in value),
        gr.DataFrame(visible="Show Metrics" in value),
    )


def handle_model_change(value: str, options: list[str]):
    compress_only = value == "Compress only"
    return (
        gr.HighlightedText(visible="Show Compressed Prompt" in options or compress_only),
        gr.Column(visible=not compress_only),
    )


def update_label(content: str, component: gr.Textbox | gr.HighlightedText):
    words = len(content.split())
    new_label = component.label.split(" (")[0] + (f" ({words} words)" if words else "")
    return (
        gr.Textbox(label=new_label)
        if isinstance(component, gr.Textbox)
        else gr.HighlightedText(label=new_label, elem_classes="no-content" if not words else "")
    )


def shuffle_and_flatten(original: dict[str, object], compressed: dict[str, object]):
    responses = [original, compressed]
    shuffle(responses)
    return [x for xs in responses for x in xs.values()]


def get_message(response: dict):
    if "choices" in response:
        return response["choices"][0]["message"]["content"]
    else:
        return f'{response["status"]} - {response["error"]}'


def create_llm_response(response: Response, compressed: bool, error: bool, start: float, end: float):
    if not error:
        response = response.json()
    obj = {
        "compressed": compressed,
        "call_time": end - start,
        "error": response.text if error else None,
        "status": response.status_code if error else None,
    }
    if not error:
        obj.update(response)
    return {"text": get_message(response) if not error else response.text, "obj": obj}


def metrics_to_df(metrics: dict):
    return pd.DataFrame(metrics["data"], columns=metrics["headers"])


def prepare_flagged_data(data: pd.DataFrame):
    if not data.empty:
        data["Response A"] = data["Response A"].apply(json.loads)
        data["Response B"] = data["Response B"].apply(json.loads)
        data["flag"] = data.apply(
            lambda x: (
                x["flag"]
                if x["flag"] == "N"
                else "Compressed" if x[f"Response {x['flag']}"]["compressed"] else "Uncompressed"
            ),
            axis=1,
        )
        data["flag"] = data["flag"].apply(lambda x: "Neither" if x == "N" else x)
        data.insert(5, "Model", data["Response A"].apply(lambda x: x["model"]))
        data["Response A"], data["Response B"] = zip(
            *data.apply(
                lambda x: (
                    map(get_message, [x["Response B"], x["Response A"]])
                    if not x["Response A"]["compressed"]
                    else map(get_message, [x["Response A"], x["Response B"]])
                ),
                axis=1,
            )
        )
        data["Metrics"] = data["Metrics"].apply(lambda x: metrics_to_df(json.loads(x)).to_html(index=False))
    data = data.rename(
        columns={"Response A": "Compressed", "Response B": "Uncompressed", "username": "user", "timestamp": "time"}
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
