
import shutil

from modeling_gpt2 import convert_gpt2_checkpoint_to_pytorch

_M_PATH = './models/mega-bert-tok/'

if __name__ == '__main__':
    convert_gpt2_checkpoint_to_pytorch(
        './models/mega-bert-tok-tf/model.ckpt-100000', 
        './models/mega-bert-tok-tf/mega.json', 
        _M_PATH
    )

