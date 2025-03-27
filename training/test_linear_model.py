import torch
import torchvision
import matplotlib.pyplot as plt

from utils import data_loader
from models import linear_model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_dim = 784
    output_dim = 10

    epoch = 100
    batch_size = 256
    hidden_dim = 103
    weight_scale = (2 / input_dim) ** 0.5  # inicjalizacji He
    decay = 1e-3

    # momentum
    beta = 0.95
    lr = 0.5
    l2_reg = 0
    dropout_p = 0
    mode = 'test'

    model = linear_model.Three_Leyer_Model(input_dim, hidden_dim,hidden_dim , output_dim, weight_scale, l2_reg, mode, device=device)

    model.load()

    train_loader, val_loader, test_loader = data_loader.get_dataloaders(batch_size=batch_size)

    epoch_test_loss_sum = 0
    epoch_test_correct = 0
    epoch_test_total = 0

    for img, y in test_loader:
        X = img.reshape(img.shape[0], -1).to(device).T
        y = y.to(device)
        z1, h1, z2, y_pred, mask_forw_drop = model.forward(X, dropout_p)
        loss = model.loss(y_pred, y)

        epoch_test_loss_sum += loss.item() * y.size(0)
        prediction = model.get_predictions(y_pred)
        epoch_test_correct += (prediction == y).sum().item()
        epoch_test_total += y.size(0)

    # Wyliczenie średnich wyników dla epoki
    avg_epoch_test_loss = epoch_test_loss_sum / epoch_test_total
    avg_epoch_test_accuracy = epoch_test_correct / epoch_test_total
    print(f" test Loss: {avg_epoch_test_loss}, test Accuracy: {avg_epoch_test_accuracy}")




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
