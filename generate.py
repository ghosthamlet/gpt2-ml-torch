
import sys
import argparse

from config import MODEL_PATH

from transformers import GPT2LMHeadModel, BertTokenizer, pipeline


def generate(prompt, model_path, n_seq, max_len, no_gpu=False, **kwargs):
    model, info = GPT2LMHeadModel.from_pretrained(model_path, output_loading_info=True)
    tokenizer = BertTokenizer.from_pretrained(model_path) 

    gpu = -1 if no_gpu else 0
    nlp = pipeline('text-generation', 
            model=model, tokenizer=tokenizer, device=gpu)
    res = nlp(
        prompt, 
        num_return_sequences=n_seq, 
        max_length=max_len, 
        do_sample=True,
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

     
