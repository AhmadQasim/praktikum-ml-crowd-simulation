import torch
from data.loader import data_loader
from evaluate_model import get_generator
from utils import animate_trajectories
from collections import namedtuple

checkpoint = torch.load('/home/ahmad/praktikum/praktikum_ml_crowd/final_project/trained_models/third_model.pt')
generator = get_generator(checkpoint)
args = namedtuple("args", checkpoint['args'].keys())(*checkpoint['args'].values())
animate_trajectories(data_loader(args,
                                 "/home/ahmad/praktikum/praktikum_ml_crowd/final_project/datasets/eth/val")[-1],
                     generator,
                     args)
