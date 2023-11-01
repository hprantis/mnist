import torch
import torchvision
from torchvision import datasets
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from helper_functions import MNISTModel, test_step, train_step, make_predictions, eval_model
import random
from torchmetrics import ConfusionMatrix, Accuracy
from mlxtend.plotting import plot_confusion_matrix

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
print(torch.version.__version__)
print(torch.version.cuda)


train_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=torchvision.transforms.ToTensor(),
    target_transform=None
)
test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=torchvision.transforms.ToTensor(),
    target_transform=None
)

print(len(train_data), len(test_data))

class_names = train_data.classes
print(class_names)
"""
image, label = train_data[0]

plt.imshow(image.squeeze())
plt.title(label)
plt.show()
"""
# dataloader
BATCH_SIZE = 32

train_dataloader = DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_dataloader = DataLoader(
    dataset=test_data,
    batch_size=BATCH_SIZE,
    shuffle=False
)

model = MNISTModel(
    input_shape=1,
    hidden_units=32,
    output_shape=len(class_names)
).to(device)

loss_fn = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1)

### TRAIN MODEL ###

epochs = 16
for epoch in range(epochs):
    print(f"Epoch: {epoch}")
    train_step(
        model=model,
        data_loader=train_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        accuracy_fn=Accuracy(task="multiclass", num_classes=len(class_names)).to(device),
        device=device
    )

    test_step(
        model=model,
        data_loader=test_dataloader,
        loss_fn=loss_fn,
        accuracy_fn=Accuracy(task="multiclass", num_classes=len(class_names)).to(device),
        device=device
    )

# get results
model_results = eval_model(
    model=model,
    data_loader=test_dataloader,
    loss_fn=loss_fn,
    accuracy_fn=Accuracy(task="multiclass", num_classes=len(class_names)),
    device=device
)

print(model_results)

# visualize results
test_samples = []
test_labels = []
for sample, label in random.sample(list(test_data), k=9):
    test_samples.append(sample)
    test_labels.append(label)

pred_probs = make_predictions(
    model=model,
    data=test_samples
)

pred_classes = pred_probs.argmax(dim=1)

print(pred_probs[:2])
print(pred_classes[:2])

plt.figure(figsize=(9, 9))
num_rows = 3
num_cols = 3

for i, sample in enumerate(test_samples):
    plt.subplot(num_rows, num_cols, i+1)
    plt.imshow(sample.squeeze())
    pred_label = class_names[pred_classes[i]]
    truth_label = class_names[test_labels[i]]

    title_text = f"Pred: {pred_label} | Truth: {truth_label}"
    plt.title(title_text)
    plt.axis(False)

plt.show()

### predictions ###
y_preds = []
model.eval()
with torch.inference_mode():
    for X, y in test_dataloader:
        X, y = X.to(device), y.to(device)
        y_logit = model(X)
        y_pred = torch.softmax(y_logit.squeeze(), dim=0).argmax(dim=1)
        y_preds.append(y_pred.cpu())

y_pred_tensor = torch.cat(y_preds)

### confusion matrix ###
confmat = ConfusionMatrix(num_classes=len(class_names), task="multiclass")
confmat_tensor = confmat(preds=y_pred_tensor,
                         target=test_data.targets)

fig, ax = plot_confusion_matrix(
    conf_mat=confmat_tensor.numpy(),
    class_names=class_names,
    figsize=(10, 7)
)

plt.show()
