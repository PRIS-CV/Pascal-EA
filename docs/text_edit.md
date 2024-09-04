# Generate and Edit Text Descriptions

## Generate text descriptions
We employ the [LLaVA1.5](https://huggingface.co/llava-hf/llava-1.5-7b-hf) to generate caption of samples in each datasets. Please run the following command.
```shell
python text_edit/generator.py --data-root $your_data_root --image-dir $your_image_dir --text-save-path $your_text_save_path
```

Here is an example:
```shell
python text_edit/generator.py --data-root /data/coco_stuff164k --image-dir images/val2017 --meta-file-path meta_val2017.json --text-save-path captions/val2017.json 
```

At last, captions will be store in your specified ```--text--save--path``` in dict format like:
```none
{"000000000285.jpg": "a close up of a brown bear sitting in a grassy field with its mouth open and its eyes looking up at the camera.", ...}
```

## Editing text descriptions 
We employ [LLama3](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) to manipulate text descriptions, where only descriptions of some attributes are changed. Please run the following command. 
```shell
python text_edit/editor.py --data-root $your_data_root --text-file-path $your_text_file_path --meta-file-path $your_meta_file_path --save-file-path $your_save_file_path --type $type
```

Here is an example:
```shell
python text_edit/editor.py --data-root /data/coco_stuff164k --text-file-path captions/val2017.json --meta-file-path meta_val2017.json --save-file-path captions/val2017_color_edit.json --type color
```

At last, the text descriptions after editing attributes specified by variable ```--type``` will be stored like:
```none
{
    "000000000285.jpg": [
        "a blue bear",
        "a green bear",
        "a yellow bear",
        "a red bear",
        "a purple bear",
        "a pink bear",
        "a orange bear",
        "a black bear",
        "a white bear",
        "a gray bear",
        "a silver bear",
        "a golden bear",
        "a turquoise bear",
        "a lavender bear",
        "a peach bear",
        "a beige bear",
        "a tan bear"
    ], ...
}
```