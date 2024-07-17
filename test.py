from llmlingua import PromptCompressor

llm_lingua = PromptCompressor(
    model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
    use_llmlingua2=True,
    device_map="mps"
)
compressed_prompt = llm_lingua.compress_prompt("The meeting was scheduled for 2:30 PM, but it started at 3:00 PM. Why?", rate=0.5)

print(compressed_prompt)