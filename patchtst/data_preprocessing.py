import argparse
import numpy as np
import pandas as pd


###################################
# Data preprocessing for PatchTST #
###################################


parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=100, help="number of clients")
parser.add_argument("--f", type=str, default='data_PatchTST.csv', help="file name")
parser.add_argument("--s", type=int, default=0, help="random seed")
args = parser.parse_args()


def data_to_csv(data, indices, days, name):
    """
    Extracts power load of clients in 'indices' during days in 'days' and saves it conveniently for PatchTST use
    """
    df = data[data['horodate'].apply(lambda x: x.date()).isin(days)][data['id_client'].isin(indices)]
    df.rename(columns={'id_client': 'unique_id', 'horodate': 'ds', 'puissance_W': 'y'}).to_csv(f'data/{name}', index=False)
    return(df)


if __name__=='__main__':
    np.random.seed(args.s)

    print('importing data...')
    calendrier_challenge = pd.read_parquet("../challenge_data/calendrier_challenge.parquet")
    questionnaire = pd.read_parquet("../challenge_data/questionnaire.parquet")
    consos_challenge = pd.read_parquet("../challenge_data/consos_challenge.parquet")
    print('data imported')

    date_df = consos_challenge['horodate'].apply(lambda x: x.date()).unique()

    # Only take participating clients
    unique_id = questionnaire[questionnaire['participe_challenge'] == True]['id_client'].unique()

    # Randomly chose the given amount among those clients
    ids = np.random.choice(unique_id, args.n)
    pivot = data_to_csv(consos_challenge, ids, [date_df[i] for i in range(365, 535)], args.f)
