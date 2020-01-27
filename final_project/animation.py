import torch
from data.loader import data_loader
from evaluate_model import get_generator
from utils import *

checkpoint = torch.load('/home/ahmad/praktikum/praktikum_ml_crowd/final_project/trained_models/first_model.pt')
generator = get_generator(checkpoint)
animate_trajectories(data_loader("/home/ahmad/praktikum/praktikum_ml_crowd/final_project/datasets/eth/test")[-1], generator)
