import pathlib
import numpy as np
import pandas as pd
import torch
from torch_lr_finder import LRFinder
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import tqdm
from tqdm.notebook import tqdm
import pickle
import gc
import time

from trainer_utils.pytorchtools import EarlyStopping

def save_dict(path, name, _dict):
    with open(path/f'{name}.pickle', 'ab') as handle:
        pickle.dump(_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


class TorchTrainer():
    def __init__(self, name, model, optimizer, loss_fn, scheduler, device, args=None, **kwargs):
        self.model = model
        self.args = args
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.device = device
        self.name = name
        self.checkpoint_path = pathlib.Path(kwargs.get('checkpoint_folder', f'./models/{name}_chkpts'))
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)
        self.train_checkpoint_interval = kwargs.get('train_checkpoint_interval', 1)
        self.max_checkpoints = kwargs.get('max_checkpoints', 25)
        self.writer = SummaryWriter(f'runs/{name}')
        self.scheduler_batch_step = kwargs.get('scheduler_batch_step', False)
        self.additional_metric_fns = kwargs.get('additional_metric_fns', {})
        self.additional_metric_fns = self.additional_metric_fns.items()
        self.pass_y = kwargs.get('pass_y', False)
        self.valid_losses = {}
        self.training_losses = {}
        self.early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

    def _get_checkpoints(self, name=None):
        checkpoints = []
        for cp in self.checkpoint_path.glob('checkpoint_*'):
            checkpoint_name = str(cp).split('/')[-1]
            checkpoint_epoch = int(checkpoint_name.split('_')[-1])
            checkpoints.append((cp, checkpoint_epoch))
        checkpoints = sorted(checkpoints, key=lambda x: x[1], reverse=True)
        # self.valid_losses = pd.read_pickle(self.checkpoint_path/'valid_losses.pickle')
        return checkpoints

    def _clean_outdated_checkpoints(self):
        checkpoints = self._get_checkpoints()
        if len(checkpoints) > self.max_checkpoints:
            checkpoints = sorted(checkpoints, key=lambda x: x[1], reverse=True)
            for delete_cp in checkpoints[self.max_checkpoints:]:
                delete_cp[0].unlink()
                print(f'removed checkpoint of epoch - {delete_cp[1]}')

    def _save_checkpoint(self, epoch, valid_loss=None):
        self._clean_outdated_checkpoints()
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': [o.state_dict() for o in self.optimizer] if type(
                self.optimizer) is list else self.optimizer.state_dict(),
        }
        if self.scheduler is not None:
            checkpoint.update({
                'scheduler_state_dict': [o.state_dict() for o in self.scheduler] if type(
                    self.scheduler) is list else self.scheduler.state_dict()
            })
        if valid_loss:
            checkpoint.update({'loss': valid_loss})
        torch.save(checkpoint, self.checkpoint_path / f'checkpoint_{epoch}')
        save_dict(self.checkpoint_path, 'valid_losses', self.valid_losses)
        save_dict(self.checkpoint_path, 'training_losses', self.training_losses)
        print(f'saved checkpoint for epoch {epoch} \n')
        self._clean_outdated_checkpoints()

    def _load_checkpoint(self, epoch=None, only_model=False, name=None):
        if name is None:
            checkpoints = self._get_checkpoints()
        else:
            checkpoints = self._get_checkpoints(name)
        if len(checkpoints) > 0:
            if not epoch:
                checkpoint_config = checkpoints[0]
            else:
                checkpoint_config = list(filter(lambda x: x[1] == epoch, checkpoints))[0]
            checkpoint = torch.load(checkpoint_config[0])
            self.model.load_state_dict(checkpoint['model_state_dict'])
            if not only_model:
                if type(self.optimizer) is list:
                    for i in range(len(self.optimizer)):
                        self.optimizer[i].load_state_dict(checkpoint['optimizer_state_dict'][i])
                else:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if self.scheduler is not None:
                    if type(self.scheduler) is list:
                        for i in range(len(self.scheduler)):
                            self.scheduler[i].load_state_dict(checkpoint['scheduler_state_dict'][i])
                    else:
                        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print(f'loaded checkpoint for epoch - {checkpoint["epoch"]}')
            return checkpoint['epoch']
        return None

    def _load_best_checkpoint(self):
        if self.valid_losses:
            best_epoch = sorted(self.valid_losses.items(), key=lambda x: x[1])[0][0]
            loaded_epoch = self._load_checkpoint(epoch=best_epoch, only_model=True)

    def _step_optim(self):
        if type(self.optimizer) is list:
            for i in range(len(self.optimizer)):
                self.optimizer[i].step()
                self.optimizer[i].zero_grad()
        else:
            self.optimizer.step()
            self.optimizer.zero_grad()

    def _step_scheduler(self, valid_loss=None):
        if type(self.scheduler) is list:
            for i in range(len(self.scheduler)):
                if self.scheduler[i].__class__.__name__ == 'ReduceLROnPlateau':
                    self.scheduler[i].step(valid_loss)
                else:
                    self.scheduler[i].step()
        else:
            if self.scheduler.__class__.__name__ == 'ReduceLROnPlateau':
                self.scheduler.step(valid_loss)
            else:
                self.scheduler.step()

    def _move_to_device(self, inputs, labels, non_blocking=True):
        def move(obj, device, non_blocking=True):
            if hasattr(obj, "to"):
                return obj.to(device, non_blocking=non_blocking)
            elif isinstance(obj, tuple):
                return tuple(move(o, device, non_blocking) for o in obj)
            elif isinstance(obj, list):
                return [move(o, device, non_blocking) for o in obj]
            elif isinstance(obj, dict):
                return {k: move(o, device, non_blocking) for k, o in obj.items()}
            else:
                return obj

        inputs = move(inputs, self.device, non_blocking=non_blocking)
        if labels is not None:
            labels = move(labels, self.device, non_blocking=non_blocking)
        return inputs, labels

    def _loss_batch(self, xb, yb, optimize, pass_y, additional_metrics=None):
        xb, yb = self._move_to_device(xb, yb)
        if pass_y:
            y_pred = self.model(xb, yb)
        else:
            # print(yb.shape)
            y_pred = self.model(xb)
        # print(f'Label: {yb.size}')
        # print(f'Preduction Out: {y_pred.size}')
        loss = self.loss_fn(y_pred, yb)
        # print(loss.isnan().any())
        if additional_metrics is not None:
            additional_metrics = [fn(y_pred, yb) for name, fn in additional_metrics]
        if optimize:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self._step_optim()
        loss_value = loss.item()
        del xb
        del yb
        del y_pred
        del loss
        if additional_metrics is not None:
            return loss_value, additional_metrics
        return loss_value

    def evaluate(self, dataloader):
        self.model.eval()
        # eval_bar = tqdm(dataloader, leave=False)
        with torch.no_grad():
            loss_values = []
            for xb, yb in dataloader:
                loss_values.append(self._loss_batch(xb, yb, optimize=False, pass_y=False, additional_metrics=self.additional_metric_fns))

            if len(loss_values[0]) > 1:
                loss_value = np.mean([lv[0] for lv in loss_values])
                additional_metrics = np.mean([lv[1] for lv in loss_values], axis=0)
                additional_metrics_result = {name: result for (name, fn), result in
                                             zip(self.additional_metric_fns, additional_metrics)}
                return loss_value, additional_metrics_result
            # eval_bar.set_description("evaluation loss %.2f" % loss_value)
            else:
                loss_value = np.mean(loss_values)
                return loss_value, None

    def predict(self, dataloader, args, n_samples=1, plot_phase=False):
        self.model.eval()
        predictions_mean = []
        predictions_std = []
        y_target = []
        idx = []
        with torch.no_grad():
            for xb, yb, idx_itr in dataloader:
                xb, yb = self._move_to_device(xb, yb)
                y_pred_sample = []
                # print(f'Input Idxs: {idx_itr}')
                for n in range(n_samples):
                    y_pred, idx_itr_out, hidden = self.model(xb, idx_itr)

                    # print(f'Out Idx - Sample: {n} - Idx: {idx_itr_out}')
                    y_pred_sample.append(y_pred)
                    # y_pred_sample.append(y_pred.cpu().numpy())
                y_pred_sample = torch.stack(y_pred_sample, dim=1)

                y_pred_sample_mean = torch.mean(y_pred_sample, dim=1)
                y_pred_sample_std = torch.std(y_pred_sample, dim=1)
                predictions_mean.append(y_pred_sample_mean.cpu().numpy())
                predictions_std.append(y_pred_sample_std.cpu().numpy())

                if plot_phase:
                    y_target.append(yb.cpu().numpy())
                    idx.append(idx_itr.cpu().numpy())

        idx = np.concatenate(idx)
        y_target = np.concatenate(y_target)
        y_target = y_target[np.argsort(idx)]
        predictions_mean = np.concatenate(predictions_mean)
        predictions_mean = predictions_mean[np.argsort(idx)]
        predictions_std = np.concatenate(predictions_std)
        predictions_std = predictions_std[np.argsort(idx)]

        if plot_phase:
            return (predictions_mean, predictions_std), y_target
        else:
            return (predictions_mean, predictions_std)

    # pass single batch input, without batch axis
    def predict_one(self, x, y, n_samples):
        y_pred_sample = None
        self.model.eval()
        with torch.no_grad():
            x, _ = self._move_to_device(x, y)
            for n in range(n_samples):
                y_pred = self.model(x)
                if n_samples > 1:
                    if n == 0:
                        y_pred_sample_mean = y_pred
                        #y_pred_sample = np.zeros((shape[0], n_samples, shape[1], shape[2], shape[3]))
                        #y_pred_sample[:, n] = y_pred.cpu().numpy()
                    elif n == 1:
                        y_pred_sample_mean = torch.stack([y_pred, y_pred_sample_mean], dim=0)
                        y_pred_sample_std = torch.std(y_pred_sample_mean, dim=0)
                        y_pred_sample_mean = torch.mean(y_pred_sample_mean, dim=0)
                    else:
                        y_pred_sample_mean_old = y_pred_sample_mean
                        y_pred_sample_mean = (y_pred_sample_mean * n + y_pred)/(n+1)
                        y_pred_sample_std = torch.sqrt(((n-1) * torch.square(y_pred_sample_std) + 
                                                    (y_pred - y_pred_sample_mean) * (y_pred - y_pred_sample_mean_old))/n)
                        # y_pred_sample[:, n] = y_pred.cpu().numpy()
                else:
                    y_pred_sample_mean = y_pred
                    y_pred_sample_std = y_pred
                    
            y_pred_sample_mean = y_pred_sample_mean.cpu().numpy()
            y_pred_sample_std = y_pred_sample_std.cpu().numpy()
            # predictions_mean = np.mean(y_pred_sample, axis=1)
            # predictions_std = np.std(y_pred_sample, axis=1)

        return (y_pred_sample_mean, y_pred_sample_std)

    def lr_find(self, dl, optimizer=None, start_lr=1e-7, end_lr=1e-2, num_iter=200):
        if optimizer is None:
            optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-6, momentum=0.9)
        lr_finder = LRFinder(self.model, optimizer, self.loss_fn, device=self.device)
        lr_finder.range_test(dl, start_lr=start_lr, end_lr=end_lr, num_iter=num_iter)
        lr_finder.plot()

    def train(self, epochs, train_dataloader, valid_dataloader=None, resume=True, resume_only_model=False):
        start_epoch = 0
        if resume:
            loaded_epoch = self._load_checkpoint(only_model=resume_only_model)
            if loaded_epoch:
                start_epoch = loaded_epoch
        len_bar = len(train_dataloader)
        for i in range(start_epoch, start_epoch + epochs):
            time_ep = time.time()
            train_dataloader_itr = iter(train_dataloader)
            self.model.train()
            training_losses = []
            running_loss = 0
            # print(len(train_dataloader))
            # training_bar = tqdm(train_dataloader, leave=False)
            # print(f'Length of Training Bar - {len_bar}')
            for it in range(len_bar):
                xb, yb = next(train_dataloader_itr)
                loss = self._loss_batch(xb, yb, True, self.pass_y)
                # print(f'Iter: {it}, Loss: {loss}')
                running_loss += loss
                # training_bar.set_description("loss %.4f" % loss)
            self.writer.add_scalar('training loss', running_loss / len_bar, i * len_bar)
            # print(f'Appending to Training Losses for Iter: {it}')
            training_losses.append(running_loss / len_bar)
            # running_loss = 0
            if self.scheduler is not None and self.scheduler_batch_step:
                self._step_scheduler()

            print(f'Training loss at epoch {i + 1} - {np.mean(training_losses)}')
            if valid_dataloader is not None:
                valid_loss, additional_metrics = self.evaluate(valid_dataloader)
                self.writer.add_scalar('validation loss', valid_loss, i)
                # if additional_metrics is not None:
                #     print(additional_metrics)
                print(f'Valid loss at epoch {i + 1} - {valid_loss}')
                self.training_losses[i + 1] = np.mean(training_losses)
                self.valid_losses[i + 1] = valid_loss

            if self.scheduler is not None and not self.scheduler_batch_step:
                self._step_scheduler(valid_loss)
            if (i + 1) % self.train_checkpoint_interval == 0:
                self._save_checkpoint(i + 1)
            print(f'Time per epoch: {time.time() - time_ep} seconds')
            del xb
            del yb

            self.early_stopping(valid_loss)
            if self.early_stopping.early_stop:
                print("Early stopping")
                break
