
import shutil
import argparse

from config import MODEL_PATH, MODEL_TF, CONFIG_TF

from modeling_gpt2 import convert_gpt2_checkpoint_to_pytorch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default=MODEL_PATH, type=str, required=False, help='Pytorch model path')
    parser.add_argument('--model_tf', default=MODEL_TF, type=str, required=False, help='Tensorflow model name')
    parser.add_argument('--config_tf', default=CONFIG_TF, type=str, required=False, help='Tensorflow model config')
    args = parser.parse_args()

    convert_gpt2_checkpoint_to_pytorch(
        args.model_tf, 
        args.config_tf, 
        args.model_path
    )

