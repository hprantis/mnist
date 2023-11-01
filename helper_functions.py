
import torch
from torch import nn
from tqdm.auto import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"


def print_time_train(start: float, end: float, device):  # device: torch.device = None
    """Prints difference between start and end time"""
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time


def eval_model(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               accuracy_fn,
               device):  # : torch.device = device
    """Returns a dictionary containing the results of the model predicting on data_loader."""
    loss, acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in tqdm(data_loader):  # for X, y in data_loader
            # device agnostoic data
            X, y = X.to(device), y.to(device)
            # make predictions
            y_pred = model(X)

            # accumulate the loss and acc values per batch
            loss += loss_fn(y_pred, y)
            #acc += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))
            acc += accuracy_fn(y, y_pred.argmax(dim=1))

        # scale the loss and acc to find avg per batch
        loss /= len(data_loader)
        acc /= len(data_loader)

    return {"model_name": model.__class__.__name__,  # only works when model was created with a class
            "model_loss": loss.item(),
            "model_acc": acc}


def train_step(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device):  #: torch.device = device
    """Performs a training step with model trying to learn on data_loader"""
    train_loss, train_acc = 0, 0

    # put model into training mode
    model.train()

    # loop through training batches
    for batch, (X, y) in enumerate(data_loader):  # X = image, y = label
        # put data on target device
        X, y = X.to(device), y.to(device)

        # 1. forward pass - outputs the raw logits from the model
        y_pred = model(X)

        # 2. calculate loss and acc per batch
        loss = loss_fn(y_pred, y)
        train_loss += loss  # accumulate train loss
        #train_acc += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))  # go from logits to prediction labels
        train_acc += accuracy_fn(y, y_pred.argmax(dim=1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # divide total train loss and acc by length of train dataloader
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f"Train loss: {train_loss:.5f} | Train acc: {train_acc:.2f}%")


def test_step(model: torch.nn.Module,
              data_loader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              accuracy_fn,
              device):  #: torch.device = device
    """Performs a testing loop step on model going over data_loader"""
    test_loss, test_acc = 0, 0
    model.eval()

    # turn on inference mode context manager
    with torch.inference_mode():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)

            # 1. forward pass
            test_pred = model(X)

            # 2. calculate loss
            test_loss += loss_fn(test_pred, y)

            # 3. accuracy
            #test_acc += accuracy_fn(y_true=y, y_pred=test_pred.argmax(dim=1))  # logits to prediction labels
            test_acc += accuracy_fn(y, test_pred.argmax(dim=1))  # logits to prediction labels

        # calculate the test loss average per batch
        test_loss /= len(data_loader)

        # calculate the test acc average per batch
        test_acc /= len(data_loader)

    print(f"Test loss: {test_loss:.4f} | Test acc: {test_acc:.4f}\n")


def make_predictions(model: torch.nn.Module,
                     data: list,
                     device: torch.device = device):
    pred_probs = []
    model.to(device)
    model.eval()
    with torch.inference_mode():
        for sample in data:
            # prepare the sample - add a batch dimension and pass to target device
            sample = torch.unsqueeze(sample, dim=0).to(device)

            # forward pass - model outputs raw logits
            pred_logit = model(sample)

            # get prediction probability - logit -> prediction probability
            pred_prob = torch.softmax(pred_logit.squeeze(), dim=0)

            pred_probs.append(pred_prob.cpu())

    # stack the pred_probs to turn list into a tensor
    return torch.stack(pred_probs)


class MNISTModel(nn.Module):
    def __init__(self, input_shape: int,
                 hidden_units: int,
                 output_shape: int):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units * 14*14,
                      out_features=output_shape)
        )

    def forward(self, x):
        return self.classifier(self.conv_block(x))


class MNISTMode1(nn.Module):
    """Model architecture that replicates the TinyVGG model from
    cnn explainer website"""

    def __init__(self, input_shape: int,
                 hidden_units: int,
                 output_shape: int):
        super().__init__()
        # create a convolutional layer
        # Extractor 1
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,  # could be a tuple (3, 3)  # this is
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))  # this can also be a tuple or digit
        )
        # Extractor 2
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units * 7 * 7,
                      out_features=output_shape)
        )

    # printing shapes through the forward pass is a way to see shapes output of model
    def forward(self, x):
        x = self.conv_block_1(x)
        #print(x.shape)
        x = self.conv_block_2(x)
        #print(x.shape)
        x = self.classifier(x)
        return x

