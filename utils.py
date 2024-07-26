import json

import gradio as gr
import pandas as pd


def create_metrics_df(result=None):
    df = pd.DataFrame(
        {
            "Original / Compressed": (
                [f'{result["origin_tokens"]} / {result["compressed_tokens"]} (tokens)'] if result else [""]
            ),
            "Ratio": [result["ratio"]] if result else [""],
            "Rate": [result["rate"]] if result else [""],
            "Saving": [result["saving"].split(", Saving ")[1]] if result else [""],
            "Compression": [""],
            "End-to-end Latency": [""],
            "End-to-end Latency Compressed (Speedup)": [""],
        }
    )
    return df


def activate_button(*values):
    return gr.Button(interactive=any(bool(value) for value in values))


def handle_ui_options(value: list[str]):
    show_prompt = "Show Separate Context Field" in value
    return (
        gr.Textbox(visible=True) if show_prompt else gr.Textbox(visible=False, value=None),
        gr.Textbox(label="Context" if show_prompt else "Prompt"),
        gr.Textbox(visible="Show Compressed Prompt" in value),
        gr.DataFrame(visible="Show Metrics" in value),
    )


def update_label(content: str, textbox: gr.Textbox):
    return gr.Textbox(label=textbox.label.split(" (")[0] + f" ({len(content.split())} words)")


def flatten(xss):
    return [x for xs in xss for x in xs]


def get_message(response):
    if "choices" in response:
        return response["choices"][0]["message"]["content"]
    else:
        return f'{response["status"]} - {response["error"]}'


def create_llm_response(response, compressed, error, start, end):
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
    return [get_message(response) if not error else response.text, obj]


def metrics_to_df(metrics):
    return pd.DataFrame(metrics["data"], columns=metrics["headers"])


def prepare_flagged_data(data):
    data["Response A"] = data["Response A"].apply(json.loads)
    data["Response B"] = data["Response B"].apply(json.loads)
    data["flag"] = data.apply(
        lambda x: (
            x["flag"] if x["flag"] == "N" else "Compressed" if x[f"Response {x['flag']}"]["compressed"] else "Uncompressed"
        ),
        axis=1,
    )
    data["flag"] = data["flag"].apply(lambda x: "Neither" if x == "N" else x)
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
    data = data.rename(
        columns={"Response A": "Compressed", "Response B": "Uncompressed", "username": "user", "timestamp": "time"}
    )
    data["Metrics"] = data["Metrics"].apply(lambda x: metrics_to_df(json.loads(x)).to_html(index=False))
    return data.iloc[::-1].to_html(table_id="table")
