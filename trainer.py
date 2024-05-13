import abc
import os
import sys
import tqdm
import torch
import bsseval

from torch.utils.data import DataLoader
from typing import Callable, Any
from pathlib import Path
from helpers.train_results import BatchResult, EpochResult, FitResult
from config import ANGLE_RES, NFFT, HOP_LENGTH


class Trainer(abc.ABC):
    """
    A class abstracting the various tasks of training models.

    Provides methods at multiple levels of granularity:
    - Multiple epochs (fit)
    - Single epoch (train_epoch/test_epoch)
    - Single batch (train_batch/test_batch)
    """

    def __init__(self, model, loss_fn, optimizer, device='cpu'):
        """
        Initialize the trainer.
        :param model: Instance of the model to train.
        :param loss_fn: The loss function to evaluate with.
        :param optimizer: The optimizer to train with.
        :param device: torch.device to run training on (CPU or GPU).
        """
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        model.to(self.device)

    def fit(self, dl_train: DataLoader, dl_test: DataLoader,
            num_epochs, checkpoints: str = None,
            early_stopping: int = None,
            print_every=1, post_epoch_fn=None, **kw) -> FitResult:
        """
        Trains the model for multiple epochs with a given training set,
        and calculates validation loss over a given validation set.
        :param dl_train: Dataloader for the training set.
        :param dl_test: Dataloader for the test set.
        :param num_epochs: Number of epochs to train for.
        :param checkpoints: Whether to save model to file every time the
            test set accuracy improves. Should be a string containing a
            filename without extension.
        :param early_stopping: Whether to stop training early if there is no
            test loss improvement for this number of epochs.
        :param print_every: Print progress every this number of epochs.
        :param post_epoch_fn: A function to call after each epoch completes.
        :return: A FitResult object containing train and test losses per epoch.
        """
        actual_num_epochs = 0
        train_loss, train_acc, test_loss, test_acc = [], [], [], []

        best_loss = None
        epochs_without_improvement = 0

        checkpoint_filename = None
        if checkpoints is not None:
            checkpoint_filename = f'{checkpoints}.pt'
            Path(os.path.dirname(checkpoint_filename)).mkdir(exist_ok=True)
            if os.path.isfile(checkpoint_filename):
                print(f'*** Loading checkpoint file {checkpoint_filename}')
                saved_state = torch.load(checkpoint_filename,
                                         map_location=self.device)
                best_loss = saved_state.get('best_loss', best_loss)
                epochs_without_improvement =\
                    saved_state.get('ewi', epochs_without_improvement)
                print(f"*** best_loss={best_loss:.3g} ewi={epochs_without_improvement}")
                self.model.load_state_dict(saved_state['model_state'])

        for epoch in range(num_epochs):
            save_checkpoint = True
            verbose = False  # pass this to train/test_epoch.
            if epoch % print_every == 0 or epoch == num_epochs - 1:
                verbose = True
            self._print(f'--- EPOCH {epoch+1}/{num_epochs} ---', verbose)

            # TODO: Train & evaluate for one epoch
            # - Use the train/test_epoch methods.
            # - Save losses and accuracies in the lists above.
            # - Implement early stopping. This is a very useful and
            #   simple regularization technique that is highly recommended.
            # ====== YOUR CODE: ======
            train_result = self.train_epoch(dl_train, **kw)
            my_mean = lambda t: sum(t) / len(t)
            current_train_loss = my_mean(train_result.losses)
            train_loss.append(current_train_loss)
            train_acc.append(train_result.accuracy.item())
            test_result = self.test_epoch(dl_test, **kw)
            current_test_loss = my_mean(test_result.losses)
            test_loss.append(current_test_loss)
            test_acc.append(test_result.accuracy.item())

            was_improved = best_loss is None or best_loss >= current_train_loss
            if was_improved:
                best_loss = current_train_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if early_stopping is not None and early_stopping <= epochs_without_improvement:
                self._print(f'--- Early Stopping! (got {epochs_without_improvement} epochs without improvement)', verbose)
                break
            # ========================

            # Save model checkpoint if requested
            if was_improved and save_checkpoint and checkpoint_filename is not None:
                saved_state = dict(best_loss=best_loss,
                                   ewi=epochs_without_improvement,
                                   model_state=self.model.state_dict())
                torch.save(saved_state, checkpoint_filename)
                print(f'*** Saved checkpoint {checkpoint_filename} '
                      f'at epoch {epoch+1}')

            if post_epoch_fn:
                post_epoch_fn(epoch, train_result, test_result, verbose)

        return FitResult(actual_num_epochs,
                         train_loss, train_acc, test_loss, test_acc)

    def train_epoch(self, dl_train: DataLoader, **kw) -> EpochResult:
        """
        Train once over a training set (single epoch).
        :param dl_train: DataLoader for the training set.
        :param kw: Keyword args supported by _foreach_batch.
        :return: An EpochResult for the epoch.
        """
        # self.model.train(True)  # set train mode
        return self._foreach_batch(dl_train, self.train_batch, **kw)

    def test_epoch(self, dl_test: DataLoader, **kw) -> EpochResult:
        """
        Evaluate model once over a test set (single epoch).
        :param dl_test: DataLoader for the test set.
        :param kw: Keyword args supported by _foreach_batch.
        :return: An EpochResult for the epoch.
        """
        # self.model.train(False)  # set evaluation (test) mode
        return self._foreach_batch(dl_test, self.test_batch, **kw)

    @abc.abstractmethod
    def train_batch(self, batch) -> BatchResult:
        """
        Runs a single batch forward through the model, calculates loss,
        preforms back-propagation and uses the optimizer to update weights.
        :param batch: A single batch of data  from a data loader (might
            be a tuple of data and labels or anything else depending on
            the underlying dataset.
        :return: A BatchResult containing the value of the loss function and
            the number of correctly classified samples in the batch.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def test_batch(self, batch) -> BatchResult:
        """
        Runs a single batch forward through the model and calculates loss.
        :param batch: A single batch of data  from a data loader (might
            be a tuple of data and labels or anything else depending on
            the underlying dataset.
        :return: A BatchResult containing the value of the loss function and
            the number of correctly classified samples in the batch.
        """
        raise NotImplementedError()

    @staticmethod
    def _print(message, verbose=True):
        """ Simple wrapper around print to make it conditional """
        if verbose:
            print(message)

    @staticmethod
    def _foreach_batch(dl: DataLoader,
                       forward_fn: Callable[[Any], BatchResult],
                       verbose=True, max_batches=None) -> EpochResult:
        """
        Evaluates the given forward-function on batches from the given
        dataloader, and prints progress along the way.
        """
        losses = []
        num_correct = 0
        num_samples = len(dl.sampler)
        num_batches = len(dl.batch_sampler)

        if max_batches is not None:
            if max_batches < num_batches:
                num_batches = max_batches
                num_samples = num_batches * dl.batch_size

        if verbose:
            pbar_file = sys.stdout
        else:
            pbar_file = open(os.devnull, 'w')

        pbar_name = forward_fn.__name__
        with tqdm.tqdm(desc=pbar_name, total=num_batches,
                    file=pbar_file) as pbar:
            try:
                dl_iter = iter(dl)
            except Exception as e:
                print(e)
            for batch_idx in range(num_batches):
                data = next(dl_iter)
                batch_res = forward_fn(data)

                pbar.set_description(f'{pbar_name} ({batch_res.loss:.3f})')
                pbar.update()

                losses.append(batch_res.loss)
                num_correct += batch_res.num_correct

            avg_loss = sum(losses) / num_batches
            accuracy = 100. * num_correct / num_samples
            pbar.set_description(f'{pbar_name} '
                                f'(Avg. Loss {avg_loss:.3f})')#, '
                                # f'Accuracy {accuracy:.1f})')

        return EpochResult(losses=losses, accuracy=accuracy)
        

class AudioTrainer(Trainer):
    def __init__(self, model, loss_fn, optimizer, device=None):
        super().__init__(model, loss_fn, optimizer, device)
        self.hidden = None

    def train_epoch(self, dl_train: DataLoader, **kw):
        # TODO: Implement modifications to the base method, if needed.
        # ====== YOUR CODE: ======
        # ========================
        return super().train_epoch(dl_train, **kw)

    def test_epoch(self, dl_test: DataLoader, **kw):
        # TODO: Implement modifications to the base method, if needed.
        # ====== YOUR CODE: ======
        # ========================
        return super().test_epoch(dl_test, **kw)

    def train_batch(self, batch) -> BatchResult:
        try:
            samples, ref_stft, target = batch
            samples = samples.to(self.device, dtype=torch.float)  # (B,S,V)
            target = target.to(self.device)
            seq_len = target.size(1)

            # TODO: Train the RNN model on one batch of data.
            self.optimizer.zero_grad()
            outputs = self.model(samples)
            # TODO
            # output_directions = torch.dot(outputs, ref_stft * ref_stft.T)
            # output_angle = torch.argmax(output_directions, axis=1)
            loss = self.loss_fn(outputs, target // ANGLE_RES)
            loss.backward()
            self.optimizer.step()

            if False:
                y_pred_index = torch.argmax(outputs, dim=1)
                y_target_index = torch.argmax(target, dim=1)
                num_correct = torch.sum(y_pred_index == y_target_index).float()
                num_correct /= outputs.numel()
            else:
                num_correct = torch.tensor(0)
                # inverse_spectrogram = torch.istft(ref_stft.squeeze(dim=1),
                #                        n_fft=NFFT,
                #                        win_length=NFFT,
                #                        hop_length=HOP_LENGTH).cpu()
                
                # DR, ISR, SIR, SAR = bsseval.evaluate(inverse_spectrogram, outputs.detach().cpu())

                
            # ========================

            # Note: scaling num_correct by seq_len because each sample has seq_len
            # different predictions.
            return BatchResult(loss.item(), num_correct)
        except Exception as e:
            print(e)
            print('what')
            print(e)

    def test_batch(self, batch) -> BatchResult:
        samples, ref_stft, target = batch
        samples = samples.to(self.device, dtype=torch.float)  # (B,S,V)
        ref_stft = ref_stft.to(self.device)
        target = target.to(self.device)
        seq_len = target.size(1)

        with torch.no_grad():
            outputs = self.model(samples)
            # TODO: ref
            loss = self.loss_fn(outputs, target // ANGLE_RES)
            if False:
                y_pred_index = torch.argmax(outputs, dim=1)
                y_target_index = torch.argmax(target, dim=1)
                num_correct = torch.sum(y_pred_index == y_target_index).float()
                num_correct /= outputs.numel()
            else:
                num_correct = torch.tensor(0)
            # ========================

        return BatchResult(loss.item(), num_correct)
