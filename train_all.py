from data.preprocess import load_and_preprocess_data
from model.abse_model import ABSEModel
from model.train_utils import train_model   


train_loader, test_loader = load_and_preprocess_data("keypoints")
model = ABSEModel(input_size=34, hidden_size=64, num_layers=1)

train_model(
    model,
    train_loader,
    test_loader,
    num_epochs=15,
    learning_rate=0.001,
    model_path="model/abse_model.pth"
) 