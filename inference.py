# Copyright (c) 2023-2024 DeepSeek.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

from argparse import ArgumentParser
from typing import List, Dict
import time
import torch
from transformers import AutoModelForCausalLM
import PIL.Image

from deepseek_vl2.models import DeepseekVLV2ForCausalLM, DeepseekVLV2Processor
from deepseek_vl2.serve.app_modules.utils import parse_ref_bbox


def load_pil_images(conversations: List[Dict[str, str]]) -> List[PIL.Image.Image]:
    """

    Args:
        conversations (List[Dict[str, str]]): the conversations with a list of messages. An example is :
            [
                {
                    "role": "User",
                    "content": "<image>\nExtract all information from this image and convert them into markdown format.",
                    "images": ["./examples/table_datasets.png"]
                },
                {"role": "Assistant", "content": ""},
            ]

    Returns:
        pil_images (List[PIL.Image.Image]): the list of PIL images.

    """

    pil_images = []

    for message in conversations:
        if "images" not in message:
            continue

        for image_path in message["images"]:
            pil_img = PIL.Image.open(image_path)
            pil_img = pil_img.convert("RGB")
            pil_images.append(pil_img)

    return pil_images


def main(args):

    dtype = torch.bfloat16

    # specify the path to the model
    model_path = args.model_path
    vl_chat_processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer

    vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=dtype
    )
    vl_gpt = vl_gpt.cuda().eval()

    # multiple images conversation example
    # Please note that <|grounding|> token is specifically designed for the grounded caption feature. It is not needed for normal conversations.
    conversation = [
        {
            "role": "<|User|>",
            "content": "<image>\n<image>\n<|grounding|>In the first image, an object within the red rectangle is marked. Locate the object of the same category in the second image.",
            "images": [
                "images/incontext_visual_grounding_1.jpeg",
                "images/icl_vg_2.jpeg"
            ],
        },
        {"role": "<|Assistant|>", "content": ""},
    ]

    ## single image conversation example
    ## Please note that <|ref|> and <|/ref|> are designed specifically for the object localization feature. These special tokens are not required for normal conversations.
    ## If you would like to experience the grounded captioning functionality (responses that include both object localization and reasoning), you need to add the special token <|grounding|> at the beginning of the prompt. Examples could be found in Figure 9 of our paper.
    # conversation = [
    #     {
    #         "role": "<|User|>",
    #         "content": "<image>\n<|ref|>The giraffe at the back.<|/ref|>.",
    #         "images": ["./images/visual_grounding_1.jpeg"],
    #     },
    #     {"role": "<|Assistant|>", "content": ""},
    # ]

    # multiple images/interleaved image-text
    # conversation = [
    #     {
    #         "role": "<|User|>",
    #         "content": "This is image_1: <image>\n"
    #                 "This is image_2: <image>\n"
    #                 "This is image_3: <image>\n Can you tell me what are in the images?",
    #         "images": [
    #             "images/multi_image_1.jpeg",
    #             "images/multi_image_2.jpeg",
    #             "images/multi_image_3.jpeg",
    #         ],
    #     },
    #     {"role": "<|Assistant|>", "content": ""}
    # ]


    # load images and prepare for inputs
    pil_images = load_pil_images(conversation)
    print(f"len(pil_images) = {len(pil_images)}")
    for img in pil_images:
        print(f"img.size = {img.size}")

    prepare_inputs = vl_chat_processor.__call__(
        conversations=conversation,
        images=pil_images,
        force_batchify=True,
        system_prompt=""
    ).to(vl_gpt.device, dtype=dtype)

    print("Input image spatial crop: ", prepare_inputs.images_spatial_crop)
    print("Input text length: ", sum(prepare_inputs.seq_lens) - sum(prepare_inputs.num_image_tokens))
    print("Input image tokens length: ", prepare_inputs.num_image_tokens)

    with torch.no_grad():
        ttft = 0
        if args.chunk_size == -1:
            # torch.cuda.reset_peak_memory_stats()
            inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)
            past_key_values = None
            # max_memory_allocated = torch.cuda.max_memory_allocated() / 1024 / 1024
            # print(f"ViT + Projector memory allocated: {max_memory_allocated} MB")
        else:
            torch.cuda.reset_peak_memory_stats()
            # incremental_prefilling when using 40G GPU for vl2-small
            for _ in range(5):
                torch.cuda.synchronize()
                tst = time.time()
                inputs_embeds, past_key_values = vl_gpt.incremental_prefilling(
                    input_ids=prepare_inputs.input_ids,
                    images=prepare_inputs.images,
                    images_seq_mask=prepare_inputs.images_seq_mask,
                    images_spatial_crop=prepare_inputs.images_spatial_crop,
                    attention_mask=prepare_inputs.attention_mask,
                    chunk_size=args.chunk_size
                )
                ted = time.time()
                ttft += (ted - tst)

        # profile TTFT
        if args.chunk_size == -1:
            for _ in range(5):
                torch.cuda.synchronize()
                tst = time.time()
                output = vl_gpt.forward(
                    inputs_embeds=inputs_embeds,
                    input_ids=None,
                    images=prepare_inputs.images,
                    images_seq_mask=prepare_inputs.images_seq_mask,
                    images_spatial_crop=prepare_inputs.images_spatial_crop,
                    attention_mask=prepare_inputs.attention_mask,
                    past_key_values=past_key_values,
                    use_cache=False,
                )
                torch.cuda.synchronize()
                ted = time.time()
                ttft += (ted - tst)
        print(
            "LLM TTFT: {:.6f} s for {} tokens".format(
                (ttft / 5), inputs_embeds.shape[1]
            )
        )

        if args.chunk_size == -1:
            torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        tst = time.time()
        # run the model to get the response
        outputs = vl_gpt.generate(
            # inputs_embeds=inputs_embeds[:, -1:],
            # input_ids=prepare_inputs.input_ids[:, -1:],
            inputs_embeds=inputs_embeds,
            input_ids=prepare_inputs.input_ids,
            images=prepare_inputs.images,
            images_seq_mask=prepare_inputs.images_seq_mask,
            images_spatial_crop=prepare_inputs.images_spatial_crop,
            attention_mask=prepare_inputs.attention_mask,
            past_key_values=past_key_values,

            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=512,

            do_sample=False,
            repetition_penalty=1.1,

            # do_sample=True,
            # temperature=0.4,
            # top_p=0.9,
            # repetition_penalty=1.1,

            use_cache=True,
            return_dict_in_generate=True
        )
        num_layers = len(outputs.past_key_values)
        num_elements = 2 * outputs.past_key_values[0][0].numel()
        dtype_size = 2 # bf16
        kv_cache_size = num_layers * num_elements * dtype_size / (1024 * 1024)
        print(f"KV Cache size: {kv_cache_size} MB")
        outputs = outputs.sequences
        
        torch.cuda.synchronize()
        ted = time.time()
        assert inputs_embeds.shape[1] == len(prepare_inputs.input_ids[0]), f"inputs_embeds.shape = {inputs_embeds.shape}, len(prepare_inputs.input_ids) = {len(prepare_inputs.input_ids)}"
        num_generated_tokens = len(outputs[0]) - len(prepare_inputs.input_ids[0])
        print("Decoding througput: {:.6f} tokens/s".format(num_generated_tokens / (ted - tst)))
        print(f"LLM max memory allocated: {torch.cuda.max_memory_allocated() / 1024 / 1024} MB")

        answer = tokenizer.decode(outputs[0][len(prepare_inputs.input_ids[0]):].cpu().tolist(), skip_special_tokens=False)
        print(f"{prepare_inputs['sft_format'][0]}", answer)

        vg_image = parse_ref_bbox(answer, image=pil_images[-1])
        if vg_image is not None:
            vg_image.save("./vg.jpg", format="JPEG", quality=85)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True,
                        default="deepseek-ai/deepseek-vl2",
                        help="model name or local path to the model")
    parser.add_argument("--chunk_size", type=int, default=-1,
                        help="chunk size for the model for prefiiling. "
                             "When using 40G gpu for vl2-small, set a chunk_size for incremental_prefilling."
                             "Otherwise, default value is -1, which means we do not use incremental_prefilling.")
    args = parser.parse_args()
    main(args)
