# basic utils
import pickle
import numpy as np
import tap
import os 

# deep learning stuff
import torch
from torch.nn.utils.rnn import pad_sequence
# wave2vec 
import gensim
import gensim.downloader
model = gensim.downloader.load('word2vec-google-news-300')

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

    path = data_path + '/' + 'variation_descriptions.pkl'
    # Get language instructions
    with open(path, 'rb') as f:
        descs = pickle.load(f)

    embs = []
    # use the fake tokens to get a padding mask
    fake_tokens = []
    for desc in descs:
        desc = desc.replace(',', '').split()
        tokens = torch.arange(1, len(desc) + 1)
        fake_tokens.append(tokens)
        desc_emb = torch.stack(
            [ torch.from_numpy(np.copy(model[w])) for w in desc ] )
        assert len(desc_emb) > 0, f"no emb for {desc}, emb shape{desc_emb}"
        embs.append(desc_emb)

    # pad all features to the same length (by adding 0s)
    # pad all token to the same length (by adding 0s)
    final_features = pad_sequence( embs, batch_first=True, padding_value = 0.0 )
    final_tokens = pad_sequence( fake_tokens, batch_first=True, padding_value = 0.0 )

    # creating padding mask based on 0-padded tokens
    padding_mask = get_padding_mask(final_tokens, pad_idx = 0.0)

    save_folder = save_path + '/' + 'language'
    if os.path.exists(save_folder) == False:
        os.mkdir(save_folder)

    # features and their corresponding padding mask into numpy
    with open(save_folder + '/w2v_features.pkl', 'wb') as handle:
        pickle.dump(final_features,  handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(save_folder + '/w2v_mask.pkl', 'wb') as handle:  
        pickle.dump(padding_mask, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("--"*5, f'sucess for {data_path[:-30]}', "--"*5)






