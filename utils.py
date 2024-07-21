import gradio as gr
import pandas as pd


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


def activate_button(value):
    return gr.Button(interactive=bool(value))

def update_label(content: str, textbox: gr.Textbox):
    return gr.Textbox(label=textbox.label.split(" (")[0] + f" ({len(content.split())} words)")