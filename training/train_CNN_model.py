import torch
from utils import data_loader
from models import cnn_model
from models import cnn_model_scratch
import torch.nn as nn
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 64

train_loader, val_loader, test_loader = data_loader.get_dataloaders(batch_size=batch_size)

learning_rate_adam = 0.001

learning_rate_SGD = 0.1
momentum = 0.9
weight_decay = 3e-5 # L2 w pytorch
num_filters = [16, 32]
max_pools = [0,1]
hidden = 50
num_epoch = 1

#model = cnn_model.conv_net(num_filters = num_filters, hidden = hidden).to(device)
model = cnn_model_scratch.DeepConvNet(input_dims=(1, 28, 28),
                                      num_filters=num_filters,
                                      max_pools=max_pools,
                                      num_classes=10,
                                      weight_scale="kaiming",
                                      dtype=torch.float,
                                      device=device).to(device) # nie odpalać tego
loss_function = nn.CrossEntropyLoss()
param_keys_model = [param for param in model.parameters()]
#optimizer_SGD = torch.optim.SGD(model.parameters(), lr=learning_rate_SGD, momentum=momentum, weight_decay = weight_decay)
optimizer_ADAM = torch.optim.Adam([model.parameters()[key] for key in param_keys_model], lr=learning_rate_adam, weight_decay=weight_decay)
train_loss = []
train_prediction = []

val_loss = []
val_preditction = []

test_loss = []
test_prediction = []
for epoch in range(num_epoch):
    avg_train_loss = 0
    avg_train_accuracy = 0

    avg_val_loss = 0
    avg_val_accuracy = 0

    avg_test_loss = 0
    avg_test_accuracy = 0
    for i, (images, y) in enumerate(train_loader):
        image = images.to(device)
        y_true = y.to(device)
        y_pred = model.loss(image, y_true)
        loss = loss_function(y_pred, y_true)

        #optimizer_SGD.zero_grad()
        optimizer_ADAM.zero_grad()
        loss.backward()
        #optimizer_SGD.step()
        optimizer_ADAM.step()

        with torch.no_grad():
            avg_train_loss += loss
            prediction = torch.argmax(y_pred, dim=1)
            accuracy = torch.sum(prediction == y_true) / y_true.shape[0]
            avg_train_accuracy += accuracy
        #optimizer_ADAM.step()

    with torch.no_grad():
        for i, (images, y) in enumerate(val_loader):
            image = images.to(device)
            y_true = y.to(device)

            y_pred = model.loss(image)
            loss = loss_function(y_pred, y_true)

            avg_val_loss += loss
            prediction = torch.argmax(y_pred, dim=1)
            accuracy = torch.sum(prediction == y_true) / y_true.shape[0]
            avg_val_accuracy += accuracy

    with torch.no_grad():
        for i, (images, y) in enumerate(test_loader):
            image = images.to(device)
            y_true = y.to(device)

            y_pred = model.loss(image)
            loss = loss_function(y_pred, y_true)

            avg_test_loss += loss
            prediction = torch.argmax(y_pred, dim=1)
            accuracy = torch.sum(prediction == y_true) / y_true.shape[0]
            avg_test_accuracy += accuracy

    avg_train_loss /= len(train_loader)
    train_loss.append(avg_train_loss)

    avg_train_accuracy /= len(train_loader)
    train_prediction.append(avg_train_accuracy)

    avg_val_loss /= len(val_loader)
    val_loss.append(avg_val_loss)

    avg_val_accuracy /= len(val_loader)
    val_preditction.append(avg_val_accuracy)

    avg_test_loss /= len(test_loader)
    test_loss.append(avg_test_loss)

    avg_test_accuracy /= len(test_loader)
    test_prediction.append(avg_test_accuracy)

    print(f"epoch {epoch}, train_loss {avg_train_loss:.4f}, train_accuracy {avg_train_accuracy:.4f}, val_loss {avg_val_loss:.4f}, val_accuracy {avg_val_accuracy:.4f}, TEST LOSS: {avg_test_loss}, TEST_ACCURACY: {avg_test_accuracy}")

train_prediction_tensor = torch.tensor(train_prediction).cpu().numpy()
train_loss_tensor = torch.tensor(train_loss).cpu().numpy()

val_prediction_tensor = torch.tensor(val_preditction).cpu().numpy()
val_loss_tensor = torch.tensor(val_loss).cpu().numpy()

test_prediction_tensor = torch.tensor(test_prediction).cpu().numpy()
test_loss_tensor = torch.tensor(test_loss).cpu().numpy()

#Conv_net.conv_net.save(model)

# Wykres strat
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(train_loss_tensor, label='Train Loss')
plt.plot(val_loss_tensor, label='val Loss')
plt.plot(test_loss_tensor, label='test Loss', color='red' ,  marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Wykres dokładności
plt.subplot(1, 2, 2)
plt.plot(train_prediction_tensor, label='Train Accuracy')
plt.plot(val_prediction_tensor, label='val Accuracy')
plt.plot(test_prediction_tensor, label='test Accuracy', color='red',  marker='o')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


