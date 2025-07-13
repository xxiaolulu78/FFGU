import pickle
import torch
import random
import numpy as np
from Fed_Unlearn_server import FFGU
import os
import datetime
"""Step 0. Initialize Federated Unlearning parameters"""

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
# ids = [0, 1, 2, 3, 4, 5, 6, 7]


class Arguments():
    def __init__(self):
        # Federated Learning Settings
        self.N_total_client = 128
        self.N_client = 10
        self.num_clients = 128
        self.use_gpu = True
        self.device = "cuda:0"

        # Federated Unlearning Settings
        self.pretrained_ckpt = "./files/ffhqrebalanced512-128.pkl"
        with open(self.pretrained_ckpt, 'rb') as f:
            snapshot_data = pickle.load(f)
            init_model = snapshot_data["G_ema"]
        self.model = init_model

        self.lr = 1e-4
        self.seed = 0
        self.fov_deg = 18.837
        self.truncation_psi = 1.0
        self.truncation_cutoff = 14
        self.iter = 800  #600

        self.angle_p = -0.2
        self.angle_y_abs = np.pi / 12
        self.sample_views = 11

        self.lambda_mse = 0.01
        self.lambda_lpips = 0.0
        self.lambda_l1 = 0.0


        self.local = True
        self.loss_local_mse_lambda = self.lambda_mse #1e-2
        self.loss_local_lpips_lambda = self.lambda_lpips #1.0
        self.loss_local_id_lambda = 0.1  #0.1
        self.loss_local_l1_lambda = self.lambda_l1
        self.adj = True
        self.loss_adj_mse_lambda = self.lambda_mse #1e-2
        self.loss_adj_lpips_lambda = self.lambda_lpips #1.0
        self.loss_adj_id_lambda = 0.1 #0.1
        self.loss_adj_l1_lambda = self.lambda_l1
        self.loss_adj_batch = 2
        self.loss_adj_lambda = 1.0
        self.loss_adj_alpha_range_min = 0
        self.loss_adj_alpha_range_max = 15

        self.glob = True
        self.loss_global_lambda = 1.0
        self.loss_global_batch = 2

        self.target_idx = 3
        self.target_cid = 0  
        self.exp = ("ffgu")
        self.inversion = "goae"
        self.target = "ffgu"
        self.target_d = 20
        self.encoder_ckpt = "./files/encoder_FFHQ.pt"


def Federated_Unlearning():
    """Step 1.Set the parameters for Federated Unlearning"""
    current_time = datetime.datetime.now()
    print("current time is:", current_time)
    FL_params = Arguments()
    seed = FL_params.seed
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
    print(FL_params.exp)
    device = torch.device("cuda:0" if FL_params.use_gpu and torch.cuda.is_available() else "cpu")
    server = FFGU(FL_params)
    # print(60 * '=')
    print("Step3. Fedearated Learning and Unlearning Training...")
    server.ffgu() 
    current_time2 = datetime.datetime.now()
    cost = current_time2 - current_time
    print("cost time:", cost)

if __name__ == '__main__':
    Federated_Unlearning()
