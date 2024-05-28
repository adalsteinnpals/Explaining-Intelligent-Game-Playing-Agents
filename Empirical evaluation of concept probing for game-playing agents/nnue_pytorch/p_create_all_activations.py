
import torch
import pandas as pd
import shelve
import features
import nnue_dataset
from tqdm import tqdm
import logging 
import numpy as np
from scipy.sparse import lil_matrix, vstack
from npy_append_array import NpyAppendArray
import click
import os
import pickle as pkl

logging.basicConfig(format='%(asctime)s — %(name)s — %(levelname)s — %(message)s', 
                    datefmt='%d-%b-%y %H:%M:%S',
                    level=logging.INFO)



def get_data(feature_set, fens):
    b = nnue_dataset.make_sparse_batch_from_fens(
        feature_set, fens, [0] * len(fens), [1] * len(fens), [0] * len(fens)
    )
    return b

def get_single_activation(model, forward_mode, feature_set, fens):

    model.forward_mode = forward_mode

    b = get_data(feature_set, fens)
    
    (
        us,
        them,
        white_indices,
        white_values,
        black_indices,
        black_values,
        outcome,
        score,
        psqt_indices,
        layer_stack_indices,
    ) = b.contents.get_tensors("cuda")
    out = model.forward(
        us,
        them,
        white_indices,
        white_values,
        black_indices,
        black_values,
        psqt_indices,
        layer_stack_indices,
    )
    out_numpy = out.cpu().detach().numpy()

    del out    


    return out_numpy





# indicate start index and length
@click.command()
@click.option('--length', default=10000, help='Length of dataset')
@click.option('--model_string', default="train_setting_6", help='Name of log for model')
@click.option('--model_file', default="model", help='Name of log for model')
@click.option('--ckpt_name', default="epoch=399-step=2441600.ckpt", help='Name of log for model')
@click.option('--batch_size', default=100, help='Size of batch')
@click.option('--bucket_str', default="", help='bucket string')
def main(length, model_string, model_file, ckpt_name, batch_size, bucket_str):

    print('model_string: ', model_string)
    if model_file == "model_single_bucket":
        import model_single_bucket as M
    elif model_file == "model":
        import model_get_activations as M
    else:
        raise Exception("model_file not recognized")
    
    dir_folder = f"production_models/{model_string}/activations"
    dir_name = f"{dir_folder}/{ckpt_name.split('-')[0].replace('=', '_')}{bucket_str}"

    # make dir if not exists
    if not os.path.exists(dir_folder):
        os.mkdir(dir_folder)
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    # load df
    df = pd.read_pickle(f"production_models/fen_df{bucket_str}.pkl")


    features_name = "HalfKAv2_hm^"
    feature_set = features.get_feature_set_from_name(features_name)

    nnue = M.NNUE(feature_set)
    nnue = nnue.load_from_checkpoint(f'production_models/{model_string}/checkpoints/{ckpt_name}', feature_set=feature_set)
    nnue = nnue.cuda()

    #nnue = torch.load(f'production_models/{model_string}/checkpoints/{ckpt_name}')
    #nnue.set_feature_set(feature_set)
    #nnue = nnue.cuda()

    #nnue = torch.load('logs/default/version_11/checkpoints/epoch=499-step=3051999.ckpt')

    #print(nnue.keys())
    #nnue.set_feature_set(feature_set)
    
    print('running')


    forward_modes = nnue.possible_forward_modes + [100]

    input_size = 46592


    # loop over start 
    for start in tqdm(range(0, df.shape[0], length), desc="start"):
        one_iteration(nnue, forward_modes, start, length, df, batch_size, input_size, feature_set, dir_name)

def one_iteration(nnue, forward_modes, start, length, df, batch_size, input_size, feature_set, dir_name):

    for forward_mode in tqdm(forward_modes, leave=False):
        
        key_name = f"layer{forward_mode}"

        if forward_mode == 10:
            arrays = lil_matrix((length, input_size), dtype=bool)
        else:
            arrays = []


        # loop over df from start to start + length in batches of batch_size
        for i, idx in enumerate(tqdm(range(start, min(start + length, df.shape[0]), batch_size), leave=False, desc=f"forward mode: {forward_mode}")):
                
                fens = df.loc[idx:idx+batch_size-1, "fen"].tolist()
    
                out = get_single_activation(nnue, forward_mode, feature_set, fens)

                #print(out.shape)
                #print(out)
    
                #print(forward_mode)
                #print(out)
                #print(out.shape)
    
                if forward_mode == 10:
                    # convert out to scripy sparse matrix
                    arrays[i:i+batch_size] = out
    
    
    
    
                    del out 
    
                
    
                else:
                    arrays.append(out)

        if forward_mode != 10:
            # concatenate arrays
            arrays = np.concatenate(arrays, axis=0)

            #print(arrays.shape)






        if 1:
            if start == 0:
                # pickle arrays
                with NpyAppendArray(f"{dir_name}/{key_name}.npy", delete_if_exists=True) as npaa:
                    npaa.append(arrays)
            else:
                with NpyAppendArray(f"{dir_name}/{key_name}.npy", delete_if_exists=False) as npaa:
                    npaa.append(arrays)

            del arrays

if __name__ == "__main__":
    main()
