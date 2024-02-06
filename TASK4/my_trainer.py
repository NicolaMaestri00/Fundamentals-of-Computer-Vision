# TUWIEN - WS2023 CV: Task4 - Mask Classification using CNN
# *********+++++++++*******++++INSERT GROUP NO. HERE
import os
from tqdm import tqdm
import numpy as np
import torch
import matplotlib.pyplot as plt
from my_model import MaskClassifier
from my_datamodule import DataModule

class Trainer:

    def __init__(self, model: MaskClassifier, datamodule: DataModule, gpu=False):
        """
        Initializes the Trainer.

        Args:
        - model (MaskClassifier): PyTorch model.
        - datamodule (DataModule): Training, validation, test data.
        - gpu (bool): Determines if the model should be trained on GPU or CPU.
        """

        self.model = model
        self.datamodule = datamodule
        self.gpu = gpu
        self.criterion = torch.nn.BCELoss()
        self.best_acc = 0
        self.history = {}

    def fit(self, epochs: int = 10, lr: float = 1e-4):
        """
        Trains the model on the training dataset.

        Args:
        - epochs (int): Number of passes through the entire training dataset.
        - lr (float): Learning rate.
        """

        self.clear_logging()  # Clean history if fit was already called

        if self.gpu:
            self.model = self.model.cuda()
            self.criterion = self.criterion.cuda()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            self.model.train()  # Set model in train state (e.g. dropouts active)

            pbar = tqdm(self.datamodule.train_dataloader())
            pbar.set_description('Epoch {}'.format(epoch))
            losses = []
            accs = []
            for imgs, labels in pbar:
                if self.gpu:
                    imgs = imgs.cuda()
                    labels = labels.cuda()

                imgs.requires_grad = True
                preds = self.model(imgs)

                loss = self.criterion(preds, labels)
                batch_accuracy = (torch.round(preds) ==
                                  labels).sum() / preds.size(0)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss = loss.cpu()
                batch_accuracy = batch_accuracy.cpu()

                # Log loss and accuracy
                losses.append(loss.detach().numpy())
                accs.append(batch_accuracy.detach().numpy())

            epoch_loss, epoch_acc = np.mean(losses), np.mean(accs)
            self.log("train_loss", epoch_loss)
            self.log("train_accuracy", epoch_acc)
            loss, acc = self.validate()
            print(f'Epoch {epoch} Training: Loss: {epoch_loss} Accuracy: {epoch_acc}\n' +
                  f'Epoch {epoch} Validation: Loss: {loss} Accuracy: {acc}')

            if self.best_acc < acc:
                self.save_model()

    def _eval(self, loader, name=str()):
        """
        Evaluates the model using the loader.

        Args:
        - loader: Data for evaluation.
        - name: Name under which the results are stored.
        """

        self.model.eval()
        accs, losses = [], []
        for imgs, labels in loader:
            if self.gpu:
                imgs = imgs.cuda()
                labels = labels.cuda()
            
            preds = self.model(imgs)
            loss = self.criterion(preds, labels)
            acc = (torch.round(preds) == labels).sum() / preds.size(0)

            if loss.is_cuda:
                loss = loss.cpu()
                acc = acc.cpu()

            losses.append(loss.detach().numpy())
            accs.append(acc.detach().numpy())

        loss, acc = np.mean(losses), np.mean(accs)
        self.log(f'{name}_accuracy', np.mean(accs))
        self.log(f'{name}_loss', np.mean(losses))
        return loss, acc
    
    
    def validate(self):
        """
        Validates the model using the validation dataset.

        Returns:
        - Tuple[float, float]: Validation loss and accuracy.
        """
        valloader = self.datamodule.val_dataloader()
        return self._eval(valloader, 'validation')

    def test(self, best_model=True):
        """
        Tests the model using the test dataset.

        Args:
        - best_model (bool): If True, the best model is loaded.

        Returns:
        - Tuple[float, float]: Test loss and accuracy.
        """
        testloader = self.datamodule.test_dataloader()
        if best_model:
            self.load_model()
        return self._eval(testloader, 'test')

    def log(self, key, val):
        """
        Stores the results of evaluation.

        Args:
        - key: Key under which the result is stored.
        - val: Value to be stored.
        """
        if self.history.get(key, None) is None:
            self.history[key] = []

        self.history[key].append(val)

    def clear_logging(self):
        """Clears the stored results."""
        self.history = {}

        
    def plot_performance(self, name: str='', group_no=0):
        """
        Visualizes the performance of training
        name: the name of the visualization
        group_no: your group number
        """

        self.test()
        test_loss = self.history['test_loss']
        test_acc = self.history['test_accuracy']

        train_acc = self.history['train_accuracy']
        val_acc = self.history['validation_accuracy']

        train_loss = self.history['train_loss']
        val_loss = self.history['validation_loss']

        max_acc = np.argmax(val_acc)
        epochs = range(1, len(val_acc)+1)

        fig = plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_acc, label='Training Accuracy')
        plt.plot(epochs, val_acc, label='Validation Accuracy')
        plt.vlines(max_acc, 0, val_acc[max_acc],
                   linestyles="dotted", colors="r")
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_loss, label='Training Loss')
        plt.plot(epochs, val_loss, label='Validation Loss')
        plt.vlines(max_acc, 0, val_loss[max_acc],
                   linestyles="dotted", colors="r")
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')

        fig.suptitle(str(group_no)+": test accuracy="+str(test_acc) +
                     ", loss: "+str(test_loss), fontsize=14, y=1)
        if name is not None and group_no is not None:
            plt.savefig(os.path.join(os.getcwd(), 'results', name))

        plt.plot()


    def _get_path(self):
        """Returns the path for saving the model."""
        path = f'results/best/{self.model.name}.pth'
        return path

    def save_model(self):
        """Saves the model state."""
        path = self._get_path()
        torch.save(self.model.state_dict(), path)

    def load_model(self):
        """Loads the saved model state."""
        self.model.load_state_dict(torch.load(self._get_path()))

    def predict(self, x, best_model=True):
        """
        Predicts a set of samples.

        Args:
        - x: Tensor of data samples.
        - best_model (bool): If True, the best model is loaded.

        Returns:
        - numpy.ndarray: Predicted values.
        """

        if best_model:
            self.load_model()

        self.model.eval()
        return self.model(x).round().detach().numpy()
