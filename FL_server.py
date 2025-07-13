
import torch
import torch.functional as F
import torch.nn as nn
import torch.optim as optim
import argparse
from torch.utils.data import DataLoader, Dataset
import copy
import numpy as np
import time
import os
import random
from model_initiation import model_init



class Server(object):
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.num_clients = args.num_clients
        self.clients = []
        self.selected_clients = []
        self.best_acc = 0.0

    def select_random_clients(self):
       
        all_clients = [f for f in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, f))]
        all_clients = [client for client in all_clients if client not in self.un_client]
        selected_clients = random.sample(all_clients, self.num_clients)
        self.selected_clients = selected_clients
        return selected_clients
