from llmlingua import PromptCompressor
import gradio as gr
import torch
import pandas as pd

llm_lingua = PromptCompressor(
    model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
    use_llmlingua2=True,
    device_map="mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu",
)


def compress_prompt(prompt, rate):
    result = llm_lingua.compress_prompt(prompt, rate=rate)

    return (
        result["compressed_prompt"],
        pd.DataFrame(
            {
                "Original / Compressed Tokens": [f'{result["origin_tokens"]} / {result["compressed_tokens"]}'],
                "Ratio": [result["ratio"]],
                "Rate": [result["rate"]],
                "Saving": [result["saving"]],
            }
        ),
    )


demo = gr.Interface(
    fn=compress_prompt,
    inputs=[gr.Textbox(lines=10, label="Prompt"), gr.Slider(0.1, 1, 0.5, label="Rate")],
    outputs=[
        gr.Textbox(lines=10, label="Compressed Prompt"),
        gr.Dataframe(label="Metrics", headers=["Original / compressed tokens", "Ratio", "Rate", "Saving"], row_count=1),
    ],
    title="Prompt Compressor",
    description="Compress a prompt using LLM-Lingua.",
    allow_flagging="never",
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
)


demo.launch()
