import torch
from data.loader import data_loader
from evaluate_model import get_generator
from utils import *

checkpoint = torch.load('/Users/mm/Desktop/Data Engineering and Analytics/3. Semester/Lab Course/praktikum-ml-crowd/final_project/trained_models/first_model.pt')
generator = get_generator(checkpoint)
evaluate(data_loader, generator)


