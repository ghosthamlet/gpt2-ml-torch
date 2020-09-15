
import sys
import argparse

from gpt2_ml_torch.config import MODEL_PATH
from gpt2_ml_torch.modeling_gpt2 import GPT2LMHeadModel

from transformers import BertTokenizer, pipeline


def generate(**kwargs):
    model_path = kwargs.pop('model_path')
    model, tokenizer, info = build_model(model_path)
    return build_output(model, tokenizer, **kwargs)


def build_model(model_path):
    model, info = GPT2LMHeadModel.from_pretrained(model_path, output_loading_info=True)
    tokenizer = BertTokenizer.from_pretrained(model_path) 
    return model, tokenizer, info


def build_output(model, tokenizer, 
        prompt, n_seq=3, max_len=300, no_gpu=False, **kwargs):
    gpu = -1 if no_gpu else 0
    nlp = pipeline('text-generation', 
            model=model, tokenizer=tokenizer, 
            device=gpu)
    res = nlp(
        prompt, 
        num_return_sequences=n_seq, 
        max_length=max_len, 
        do_sample=True,
        return_dict=False, 
        **kwargs
    )

    return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, required=True, help='Prompt text')
    parser.add_argument('--model_path', default=MODEL_PATH, type=str, required=False, help='Pytorch Model path')
    parser.add_argument('--max_len', default=300, type=int, required=False, help='Max seq len')
    parser.add_argument('--n_seq', default=3, type=int, required=False, help='Generate seq numbers')
    parser.add_argument('--no_gpu', action='store_true', required=False, help='Disable gpu')
    args = parser.parse_args()

    res = generate(**args.__dict__)

    for i, v in enumerate(res):
        print()
        print('%d. %s' % (i, v['generated_text']))

     
