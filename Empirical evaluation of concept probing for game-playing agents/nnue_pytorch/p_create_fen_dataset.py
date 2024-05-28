from nnue_dataset import FenBatchProvider
from tqdm import tqdm 
import chess
import pandas as pd
import copy
from collections import Counter



def remove_pieces_from_piece_map(piece_map, pieces):
    _piece_map = copy.copy(piece_map)
    for key, value in piece_map.items():
        if str(value) in pieces:
            del _piece_map[key]

    return _piece_map




def fetch_batch(FBP):
    fens = next(FBP)
    results = []
    for i, fen in enumerate(fens):
        d = {}
        d['fen'] = fen
        board = chess.Board(fen)
        pieces = dict(Counter([str(v) for v in board.piece_map().values()]))
        d = {**d, **pieces}
        d['num_pieces'] = sum(pieces.values())
        d['bucket'] = int((sum(pieces.values()) - 1) / 4)
        d['white_to_move'] = board.turn
        results.append(d)

    return results


def bucket_is_N(d):
    return d['bucket'] == 2

def gen_fen_dataset(size = 1000000, filter = bucket_is_N):
    batch_size = 100


    FBP = FenBatchProvider('../PycharmProjects/data/training_data.binpack', True, 1, batch_size=batch_size)


    data = []
    with tqdm(total=size) as pbar:

        while len(data) < size:
            unfiltered_data = fetch_batch(FBP)
            
            if filter is not None:
                filtered_data = [d for d in unfiltered_data if filter(d)]
            else:
                filtered_data = unfiltered_data

            data += filtered_data
            pbar.update(len(filtered_data))

    df = pd.DataFrame(data).fillna(0)
    # drop duplicate fen strings
    print(df.shape)
    df = df.drop_duplicates(subset=['fen'])
    print(df.head(50))
    print(df.shape)

    return df

if __name__ == '__main__':
    df = gen_fen_dataset()

    df.to_pickle('production_models/fen_df_bucket2.pkl')

