from math import e
import os
import re
import json
import torch
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

def get_prompt(edit_type, object_cat):
    if edit_type == "material":
        return f"I want to change the material of the {object_cat} in source image. Please generate all possible target text prompts given the source text prompt describing the source image. For example, source is 'a cat', you can generate 'a wooden cat sculpture'."
    elif edit_type == "color":
        return f"I want to change the color of the {object_cat} in source image. Please generate all possible target text prompts given the source text prompt describing the source image. For example, source is 'a cat', you can generate 'a blue cat'."
    elif edit_type == "style":
        return "I want to change the image style of source images without perturbating the content. Please generate all possible target text prompts given the source text prompt describing the source image. For example, source is 'a cat', you can generate 'a watercolor cat'."
    elif edit_type == "weather":
        return "I want to change the weather or season condition of the source image. Please generate all possible target text prompts given the source text prompt describing the source image, by only changing the weather conditions, or adding a description of the weather if not already present."
    else:
        raise ValueError("support attribute types: material, color, stype, weather")
    
def main():
    parser = argparse.ArgumentParser(description='path')
    parser.add_argument('--data-root', type=str, required=True)
    parser.add_argument('--text-file-path', type=str, required=True)
    parser.add_argument('--type', type=str, default="color", choices=["color", "material", "style", "weather"])
    parser.add_argument('--meta-file-path', type=str, required=True)
    parser.add_argument('--save-file-path', type=str, required=True)
    args = parser.parse_args()

    text_file_path = os.path.join(args.data_root, args.text_file_path)
    meta_file_path = os.path.join(args.data_root, args.meta_file_path)
    edit_save_path = os.path.join(args.data_root, args.save_file_path)
    
    with open(text_file_path, "r") as f:
        captions = json.load(f)
    with open(meta_file_path, "r") as f:
        meta_info = json.load(f)
    
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map=device,)
    terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]

    img_list = list(meta_info.keys())
    caption_edited = dict()
    for img in tqdm(img_list):
        source_caption = captions[img]
        
        # chat with LLM
        instructtion = get_prompt(edit_type=args.type, object_cat=meta_info[img])
        messages = [
            {"role": "system", "content": "You are a chat bot who always appropriately respond the request."},
            {"role": "user", "content": instructtion},
            {"role": "user", "content": f"The source prompt is {source_caption}. Please just list target prompts without other thing else."}
        ]
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)
        outputs = model.generate(
            input_ids,
            max_new_tokens=256,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        response = outputs[0][input_ids.shape[-1]:]
        response_text = tokenizer.decode(response, skip_special_tokens=True)
        
        # post-process texts
        raw_response = response_text.split("\n\n")[-1]
        raw_prompts = raw_response.split("\n")
        target_prompts = list()
        for input_string in raw_prompts:
            result = re.match(r"(\d+\.|\*)\s*(.*)", input_string)
            if result:
                content = result.group(2)
                target_prompts.append(content)
        caption_edited[img] = target_prompts
    
    with open(edit_save_path, 'w') as fp:
        text = json.dumps(caption_edited, indent=4)
        fp.write(text)
    
if __name__ == "__main__":
    main()