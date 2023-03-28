# basic utils
import pickle
import numpy as np
import tap
import os 

# deep learning stuff
import torch
from torch.nn.utils.rnn import pad_sequence
import clip

# project-specific
from Networks.utils import get_padding_mask

class ArgumentParser(tap.Tap):
    data_path: str  
    save_path: str
    variations: str

args = ArgumentParser().parse_args()
variations = args.variations.split()

for variation in variations:
    data_path = args.data_path + variation
    save_path = args.save_path + variation

    args = ArgumentParser().parse_args()
    path = data_path + '/' + 'variation_descriptions.pkl'
    save_path = save_path

    # Get language instructions
    with open(path, 'rb') as f:
        desc = pickle.load(f)
    # Clip setup
    model, preprocess = clip.load("ViT-B/32")
    model.cuda().eval()

    # for a list of language descriptions, tokenize them with CLIP.
    padded_tokens = clip.tokenize( desc).cuda()

    # obtain true length (bpe-level) of each language instructions
    eos_idx = padded_tokens.argmax(dim=-1).cpu().numpy()

    with torch.no_grad():
        # note that I modified the clip code so that we retrieve token-level
        # features exclulding bos & eos instead of features of f<eos>
        # return value is sequence of vectors. len(seq) == # of words
        padded_features = model.featurized_text(padded_tokens).detach().cpu().float()

    features = []
    tokens = []
    eos_features = [] # use as a sentence level feature
    # get features excluding bos 
    for idx, text_feature in enumerate( padded_features ):
        # eos feature used by LAVA, preserve dim by slicing
        eos_feature = text_feature[eos_idx[idx]:eos_idx[idx]+1].clone().cpu()
        # feature excluding bos and eos token used by VALA
        new_feature = text_feature[1: eos_idx[idx]].clone().cpu()
        # get unpad tokens for padding mask creation
        new_token = padded_tokens[idx][1: eos_idx[idx]].clone().cpu()

        features.append(new_feature)
        tokens.append(new_token)
        eos_features.append(eos_feature)

    # pad all features to the same length (by adding 0s)
    # pad all token to the same length (by adding 0s)
    final_features = pad_sequence( features, batch_first=True, padding_value = 0.0 )
    final_tokens = pad_sequence( tokens, batch_first=True, padding_value = 0.0 )
    eos_features = torch.cat(eos_features)

    # creating padding mask based on 0-padded tokens
    padding_mask =  get_padding_mask(final_tokens, pad_idx = 0.0)

    save_folder = save_path + '/' + 'language'
    if os.path.exists(save_folder) == False:
        os.mkdir(save_folder)

    # features and their corresponding padding mask into numpy
    with open(save_folder + '/token_features.pkl', 'wb') as handle:
        pickle.dump(final_features,  handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(save_folder + '/eos_features.pkl', 'wb') as handle:
        pickle.dump(eos_features, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(save_folder + '/padding_mask.pkl', 'wb') as handle:  
        pickle.dump(padding_mask, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("--"*5, 'instruction featurization sucess', "--"*5)
    # for CLIP, change this function within the model.py
    """
    def featurized_text(self, text):
            x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

            x = x + self.positional_embedding.type(self.dtype)
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = self.ln_final(x).type(self.dtype)

            # x.shape = [batch_size, n_ctx, transformer.width]
            # take features from the eot embedding (eot_token is the highest number in each sequence)
            x = x @ self.text_projection

            return x
    """




