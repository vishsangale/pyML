# Adapted from https://github.com/pytorch/examples/blob/master/mnist/main.py
import argparse
import logging

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

from torch.optim.lr_scheduler import StepLR


learning_rate = 1.0


def plot_training_samples(loader):
    examples = enumerate(loader)
    batch_idx, (example_data, example_targets) = next(examples)
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap="gray", interpolation="none")
        plt.title("Ground Truth: {}".format(example_targets[i]))
        plt.xticks([])
        plt.yticks([])
    plt.show()


def parse_arguments():
    parser = argparse.ArgumentParser(description="MNIST example using pytoch.")
    parser.add_argument(
        "-bs",
        "--batch_size",
        dest="batch_size",
        type=int,
        default=64,
        help="Batch size for training iteration.",
    )
    parser.add_argument(
        "-s",
        "--seed",
        dest="seed",
        type=int,
        default=13,
        help="Seed value for random number generation.",
    )
    return parser.parse_args()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            current_samples = batch_idx * len(data)
            logging.info(
                f"Train Epoch: {epoch} [{current_samples:5d}/{len(train_loader.dataset)}] Loss: {loss.item():.6f}"
            )


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    nr_test_samples = len(test_loader.dataset)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= nr_test_samples

    logging.info(
        f"Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{nr_test_samples}"
    )


def main(args):
    logging.debug(f"Torch version {torch.__version__}")
    logging.debug(f"Torchvision version {torchvision.__version__}")
    torch.manual_seed(args.seed)
    logging.info(f"Seed set to {args.seed}")
    if not torch.cuda.is_available():
        logging.error("GPU not available, using CPU")
        device = torch.device("cpu")
    else:
        logging.debug(f"Using GPU {torch.cuda.get_device_name(0)}")
        device = torch.device("cuda")

    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            "data/",
            train=True,
            download=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        ),
        batch_size=args.batch_size,
        shuffle=True,
    )

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            "data/",
            train=False,
            download=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        ),
        batch_size=args.batch_size,
        shuffle=True,
    )

    # To visualize training samples
    # plot_training_samples(train_loader)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=learning_rate)

    scheduler = StepLR(optimizer, step_size=1, gamma=0.5)
    for epoch in range(1, 6):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)
        scheduler.step()

    torch.save(model.state_dict(), "data/mnist_cnn.pt")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s [%(levelname)-8s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.DEBUG,
    )
    main(parse_arguments())
