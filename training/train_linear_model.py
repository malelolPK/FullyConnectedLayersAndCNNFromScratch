import torch
import torchvision
import matplotlib.pyplot as plt

from utils import data_loader
from models import linear_model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #hiperparamters
    input_dim = 784
    output_dim = 10

    epoch = 50
    batch_size = 128
    hidden_dim_one = 125
    hidden_dim_two = 20
    weight_scale = (2 / input_dim) ** 0.5  # inicjalizacji He
    decay = 1e-3

    # momentum
    beta = 0.90
    lr = 0.35

    # weight_scale = 1e-3
    eta = 0.01
    l2_reg = 0
    dropout_p = 0

    train_loader, val_loader, test_loader = data_loader.get_dataloaders(batch_size=batch_size)
    mode = 'train'

    model = linear_model.Three_Leyer_Model(input_dim, hidden_dim_one, hidden_dim_two, output_dim, weight_scale, l2_reg, mode, device=device)

    model.initialize_velocity_momentum()
    #model.load()
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    patience = 10
    best_loss = float('inf')
    epochs_no_improve = 0

    for t in range(epoch):
        epoch_train_loss_sum = 0
        epoch_train_correct = 0
        epoch_train_total = 0


        mode = 'train'
        for i, (images, y) in enumerate(train_loader):
            X = images.reshape(images.shape[0], -1).to(device).T
            y = y.to(device)
            z1, h1, z2,h2, z3, y_pred, mask_forw_drop = model.forward(X, dropout_p)
            loss = model.loss(y_pred, y)
            grads = model.backward(X, y_pred, y, z1, h1, z2, h2, z3, mask_forw_drop)

            #model.adam(t, eta)
            #model.SGD(learning_rate_SGD)
            model.SGD_momentum(beta, lr)

            epoch_train_loss_sum += loss.item() * y.size(0)
            prediction = model.get_predictions(y_pred)
            epoch_train_correct += (prediction == y).sum().item()
            epoch_train_total += y.size(0)

        # Wyliczenie średnich wyników dla epoki
        avg_epoch_train_loss = epoch_train_loss_sum / epoch_train_total
        avg_epoch_train_accuracy = epoch_train_correct / epoch_train_total
        train_losses.append(avg_epoch_train_loss)
        train_accuracies.append(avg_epoch_train_accuracy)

        print(f"Epoch {t} Train Loss: {avg_epoch_train_loss}, Train Accuracy: {avg_epoch_train_accuracy}")

        epoch_val_loss_sum = 0
        epoch_val_correct = 0
        epoch_val_total = 0
        mode = 'val'

        with torch.no_grad():
            for i, (images, y) in enumerate(val_loader):
                X = images.reshape(images.shape[0], -1).to(device).T
                y = y.to(device)
                z1, h1, z2,h2,z3, y_pred, mask_forw_drop = model.forward(X, dropout_p)
                val_loss = model.loss(y_pred, y)

                epoch_val_loss_sum += val_loss.item() * y.size(0)
                prediction = model.get_predictions(y_pred)
                epoch_val_correct += (prediction == y).sum().item()
                epoch_val_total += y.size(0)

        # Wyliczenie średnich wyników dla epoki walidacyjnej
        avg_epoch_val_loss = epoch_val_loss_sum / epoch_val_total
        avg_epoch_val_accuracy = epoch_val_correct / epoch_val_total
        val_losses.append(avg_epoch_val_loss)
        val_accuracies.append(avg_epoch_val_accuracy)

        if best_loss > avg_epoch_val_loss:
            best_loss = avg_epoch_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        print(f"Epoch {t} Val Loss: {avg_epoch_val_loss}, Val Accuracy: {avg_epoch_val_accuracy}")

        if epochs_no_improve >= patience:
            print("Early stopping")
            break

        # time based decey
        #new_eta = model.time_based_step_decay(epoch, eta, decay)
        #eta = new_eta

        #new_lr = model.time_based_step_decay(epoch, lr, decay)
        #lr = new_lr

        # exponetial decey
        #new_eta = model.exponential_step_decay(epoch, eta, decay)
        #eta = new_eta

        new_lr = model.exponential_step_decay(epoch, lr, decay)
        lr = new_lr

    #model.save()

    # Konwersja list na tensory aby nie były w gpu
    train_accuracies_tensor = torch.tensor(train_accuracies).cpu().numpy()
    val_accuracies_tensor = torch.tensor(val_accuracies).cpu().numpy()
    train_losses_tensor = torch.tensor(train_losses).cpu().numpy()
    val_losses_tensor = torch.tensor(val_losses).cpu().numpy()

    # Wykres strat
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses_tensor, label='Train Loss')
    plt.plot(val_losses_tensor, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Wykres dokładności
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies_tensor, label='Train Accuracy')
    plt.plot(val_accuracies_tensor, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
