import torch
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

def inference(model, processer, messages):
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids,
        out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    return output_text

model_path = "Qwen/Qwen2-VL-2B-Instruct-AWQ"
torch_dtype = torch.float16
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto"
).to(torch_dtype).cuda()

min_pixels = 512*28*28
max_pixels = 1024*28*28
processor = AutoProcessor.from_pretrained(
    model_path,
    min_pixels=min_pixels,
    max_pixels=max_pixels
)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "demo.jpeg",
            },
            {"type": "text", "text": "この画像について説明してください"},
        ],
    }
]

output_text = inference(model, processor, messages)
print(output_text)