import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from learn_d_z import learn_d_z
from tqdm import tqdm




calendrier_challenge = pd.read_parquet("challenge_data/calendrier_challenge.parquet")
questionnaire = pd.read_parquet("challenge_data/questionnaire.parquet")
consos_challenge = pd.read_parquet("challenge_data/consos_challenge.parquet")
temperatures = pd.read_parquet("challenge_data/temperatures.parquet")



min_date = datetime.datetime(2009,7,15,12)
all_clients = list(set(consos_challenge["id_client"]))
day = datetime.datetime(2011,1,2)-datetime.datetime(2011,1,1)


#penalisation
n_days = 365
n_times = n_days*48
n_atoms = 15
length_atoms = 48
coef_reg = 0.5
reg = np.ones(n_times-length_atoms+1)
t_j = np.linspace(-12,12,48)
x_j = 1+t_j**2
for i in range(n_times//48-1):
    reg[48*i:48*(i+1)]=x_j
reg = reg*coef_reg


def load_signal(id_client,min_time,max_time):
    consos = consos_challenge[consos_challenge["id_client"]==id_client]
    consos = consos[consos["horodate"]>=min_time]
    return consos[consos["horodate"]<max_time]["puissance_W"]



for id_client in tqdm(all_clients):
    signal = np.array(load_signal(id_client,min_date,min_date+n_days*day))
    X = signal.reshape(1,len(signal))
    cdl = learn_d_z(X,n_atoms,48,reg=reg,verbose=0)
    spike = cdl[3]
    atoms = cdl[2]

    np.save('data_spike/'+str(id_client)+'.npy',spike)
    np.save('data_atoms/'+str(id_client)+'.npy',atoms)



















