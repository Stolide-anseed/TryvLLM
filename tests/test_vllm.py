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
        "Ты помошник в сфере кинокритики"
        "Твоя задача объективно оценивать фильмы"
        "Твой ответ должен состоять из: о чём фильм, главные герои, плюсы, минусы, итог"
        "Оцени Фильм Шрэк 1"
    ]

    outputs = llm.chat(
        prompts,
        sampling_params=sampling_params,
    )

    for output in outputs:
        print("PROMPT:", output.prompt)
        print("OUTPUT:", output.outputs[0].text)
        print("-" * 30)


if __name__ == "__main__":
    main()