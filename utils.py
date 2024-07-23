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
    show_prompt = "Show separate context field" in value
    return (
        gr.Textbox(visible=True) if show_prompt else gr.Textbox(visible=False, value=None),
        gr.Textbox(label="Context" if show_prompt else "Prompt"),
        gr.DataFrame(visible="Show Metrics" in value),
        gr.Textbox(visible="Show Compressed Prompt" in value),
    )


def update_label(content: str, textbox: gr.Textbox):
    return gr.Textbox(label=textbox.label.split(" (")[0] + f" ({len(content.split())} words)")


def flatten(xss):
    return [x for xs in xss for x in xs]


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
    return response["choices"][0]["message"]["content"] if not error else response.text, obj
