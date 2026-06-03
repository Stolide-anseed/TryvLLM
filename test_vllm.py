import os

from vllm import LLM, SamplingParams


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

    prompts = [
        "Кто такой Шрэк, объясни воровским слэнгом",
        "Как работает логистическая регрессия",
    ]

    outputs = llm.generate(
        prompts,
        sampling_params=sampling_params,
    )

    for output in outputs:
        print("PROMPT:", output.prompt)
        print("OUTPUT:", output.outputs[0].text)
        print("-" * 30)


if __name__ == "__main__":
    main()