import os

from vllm import LLM, SamplingParams

# Вызов модели для QA сессии

def main():
    llm = LLM(
        model="Qwen/Qwen3-0.6B",
        dtype="auto",
        max_model_len=1024,
        gpu_memory_utilization=0.7,
        hf_token=os.getenv("HF_TOKEN"),
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=0.9,
        max_tokens=256,
    )

    messages = [
        {
            "role": "system",
            "content": "Ты полезный ассистент. Отвечай кратко и понятно.",
        },
        {
            "role": "user",
            "content": "Объясни простыми словами, что такое KV-cache.",
        },
    ]

    outputs = llm.chat(
        messages,
        sampling_params=sampling_params,
    )


    request_output = outputs[0]
    completion = request_output.outputs[0]

    print(completion.text)
    print(f"\nPrompt tokens: {len(request_output.prompt_token_ids or [])}")
    print(f"Generated tokens: {len(completion.token_ids)}")
    print(f"Finish reason: {completion.finish_reason}")


if __name__ == "__main__":
    main()