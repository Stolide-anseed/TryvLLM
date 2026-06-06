from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    model_name = "Qwen/Qwen3-0.6B"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype='auto',
        device_map = 'auto',
    )



    messages = [
        {
            "role": "system",
            "content": (
                "Ты помощник в сфере кинокритики. "
                "Твоя задача объективно оценивать фильмы."
            ),
        },
        {
            "role": "user",
            "content": (
                "Оцени фильм Шрэк 1. Ответ должен состоять из: "
                "о чем фильм, главные герои, плюсы, минусы, итог."
            ),
        },
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # conduct text completion
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=1024
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

    try:
        # rindex finding 151668 (</think>)
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

    print("thinking content:", thinking_content)
    print("content:", content)
if __name__ == "__main__":
    main()