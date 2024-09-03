import os
import json
import torch
from tqdm import tqdm
from PIL import Image
from argparse import ArgumentParser
from transformers import BitsAndBytesConfig
from transformers import AutoProcessor, LlavaForConditionalGeneration

# Generate text description for each samples in dataset using LLaVA1.5
def main():
    parser = ArgumentParser()
    parser.add_argument('--data-root', type=str, required=True)
    parser.add_argument('--image-dir', type=str, required=True)
    parser.add_argument('--meta-file-path', type=str, required=True)
    parser.add_argument('--text-save-path', type=str, required=True)
    args = parser.parse_args()
    
    model_id = "llava-hf/llava-1.5-7b-hf"
    processor = AutoProcessor.from_pretrained(model_id)
    quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    model = LlavaForConditionalGeneration.from_pretrained(model_id, quantization_config=quantization_config, device_map="cuda")

    images_dir = os.path.join(args.data_root, args.data_root)
    text_save_path = os.path.join(args.data_root, args.text_save_path)

    with open(args.meta_file_path, 'r') as fp:
        categories = json.load(fp)

    description = dict()
    instances = sorted(categories.keys())

    for ins in tqdm(instances):
        cat = categories[ins]
        prompt = [f"USER: <image>\nPlease briefly generate caption of the {cat}\nASSISTANT:"]
        image = Image.open(os.path.join(images_dir, ins))
        
        input = processor(prompt, images=[image], padding=True, return_tensors="pt").to("cuda")
        output = model.generate(**input, max_new_tokens=200)
        
        generated_text = processor.batch_decode(output, skip_special_tokens=True)
        text = generated_text[0]
        description[ins] = text.split("ASSISTANT:")[-1]

    with open(text_save_path, 'w') as fp:
        text = json.dumps(description, indent=4)
        fp.write(text)  


if __name__ == "__main__":
    main()