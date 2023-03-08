### Solver (train and test)
import os
import time
import copy
import shutil
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.tensorboard import SummaryWriter

from recorder import Recorder
from utils import makeSubdir, logInfoWithDot, timeSince, init_model, init_optimizer

if __name__ == "__main__":
    from utils import init_model, load_model, parse_model, plot_confusion_matrix


class Solver():
    def __init__(self, model, dataloader, optimizer, scheduler, logger, writer):
        """
        Args:
            model:         Model params
            dataloader:    Dataloader params
            optimizer:     Optimizer params
            scheduler:     Scheduler params
            logger:        Logger params
            writer:        Tensorboard writer params
        """
        self.model = init_model(model)
        self.pretrained = model['pretrained']
        self.loading_epoch = model['loading_epoch']
        self.total_epochs = model['total_epochs']
        self.model_path = model['model_path']
        self.model_filename = model['model_filename']
        self.save = model['save']
        self.model_name = model['model_type']
        self.model_fullpath = str(self.model_path / self.model_filename)

        self.init_random_seed = model['manual_seed']

        # Best model
        self.is_best = False
        self.best_model = None
        self.best_acc = 0
        self.best_optim = None
        self.best_epoch = 0
        self.best_model_filepath = str(
            self.model_path / 'best_model_checkpoint.pth.tar')

        # Logger
        self.logger = logger
        # Writer: Default path is runs/CURRENT_DATETIME_HOSTNAME
        writer_fullname = writer['writer_path'] / writer['writer_filename']
        self.writer = SummaryWriter(log_dir=writer_fullname)

        if model['gpu'] and torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.model.to(self.device)
            logInfoWithDot(self.logger, "USING GPU")
        else:
            self.device = torch.device('cpu')
            logInfoWithDot(self.logger, "USING CPU")

        if model['resume']:
            load_model_path = self.model_fullpath.format(
                self.model_name, self.loading_epoch)
            checkpoint = torch.load(load_model_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            # Log information
            logInfoWithDot(self.logger, "LOADING MODEL FINISHED")

        if optimizer['weighted_loss']:
            weights = torch.tensor([1., 3.]).to(self.device)
            self.logger.info(f'weighted {weights}')
            self.criterion = nn.CrossEntropyLoss(weight=weights)
        else:
            self.criterion = nn.CrossEntropyLoss()

        # Optimizer
        self.optimizer = init_optimizer(optimizer, self.model)
        if optimizer['resume']:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # Log information
            logInfoWithDot(self.logger, "LOADING OPTIMIZER FINISHED")

        # Scheduler
        self.scheduler = None
        if scheduler['use']:
            self.scheduler = optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=scheduler['milestones'],
                gamma=scheduler['gamma'])
            # self.scheduler = optim.lr_scheduler.StepLR(
            #     self.optimizer,
            #     step_size=scheduler['step_size'],
            #     gamma=scheduler['gamma'])

        # Dataloader
        self.trainloader = dataloader['train']
        self.validloader = dataloader['valid']

        # Evaluation
        self.train_recorder = Recorder('train')
        self.test_recorder = Recorder('test')

        # Timer
        self.start_time = time.time()

    def train_one_epoch(self, ep):
        pass

    def test_one_epoch(self, ep):
        pass

    def forward(self):
        start_epoch = self.loading_epoch + 1
        total_epochs = start_epoch + self.total_epochs
        for epoch in range(start_epoch, total_epochs):

            np.random.seed(self.init_random_seed+epoch)

            self.train_one_epoch(epoch)
            self.test_one_epoch(epoch)

            if self.scheduler:
                self.scheduler.step()

            if self.save and self.is_best:
                makeSubdir(self.model_path)
                torch.save({
                    'epoch': self.best_epoch,
                    'model_state_dict': self.best_model.state_dict(),
                    'optimizer_state_dict': self.best_optim.state_dict(),
                    'best_acc': self.best_acc,
                }, self.best_model_filepath)

                # Log information
                logInfoWithDot(
                    self.logger, "SAVED MODEL: {}".format(
                        self.model_fullpath.format(self.model_name, self.best_epoch)))

        # Rename
        shutil.move(self.best_model_filepath, self.model_fullpath.format(
            self.model_name, self.best_epoch))

        # self.writer.add_graph(self.model)
        self.writer.close()
        logInfoWithDot(
            self.logger, "TRAINING FINISHED, TIME USAGE: {} secs".format(
                timeSince(self.start_time)))


class HyphalSolver(Solver):
    def train_one_epoch(self, ep, log_interval=50):
        self.model.train()

        self.train_recorder.reset()

        lr = self.optimizer.param_groups[0]['lr']

        for i, (images, labels) in enumerate(self.trainloader, 0):
            images = images.to(self.device)
            labels = labels.to(self.device)

            if not self.pretrained and self.model_name == 'GoogleNet':
                preds, aux2, aux1 = self.model(images)
                loss1 = self.criterion(preds, labels)
                loss2 = self.criterion(aux1, labels)
                loss3 = self.criterion(aux2, labels)
                loss = loss1 + 0.3 * (loss2 + loss3)
            elif self.model_name == 'Inception3':
                preds, aux = self.model(images)
                loss1 = self.criterion(preds, labels)
                loss2 = self.criterion(aux, labels)
                loss = loss1 + 0.4 * loss2
            else:
                preds = self.model(images)
                loss = self.criterion(preds, labels) # binary cross-entropy loss

            # Update
            self.train_recorder.update(preds, labels, loss.item())
            # Backward propagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if i % log_interval == 0:
                self.logger.info(
                    'Train Epoch: {} [{}/{} ({:.0f}%)]\tLearning Rate: {}\tLoss: {:.6f}\tTime Usage:{:.8}'
                    .format(ep, i * len(images), len(self.trainloader.dataset),
                            100. * i / len(self.trainloader), lr, loss.data,
                            timeSince(self.start_time)))

            if i == len(self.trainloader) - 1:
                # Calculate the new counts of false positives, false negatives, true positives, true negatives, and correct predictions
                fp = self.train_recorder.false_positive_counts
                fn = self.train_recorder.false_negative_counts
                tp = self.train_recorder.true_positive_counts
                tn = self.train_recorder.true_negative_counts
                correct = self.train_recorder.correct_counts
                total = self.train_recorder.total_counts

                # Calculate the accuracy, precision, recall, and F1 score
                accuracy = 100.0 * correct / total
                precision = 100.0 * tp / (tp + fp)
                recall = 100.0 * tp / (tp + fn)
                f1 = 100 * (2 * precision * recall / (precision + recall))
                
                self.logger.info(
                    'Loss of the network on the {0} train images: {1:.6f}'.
                    format(total, self.train_recorder.loss))
                self.logger.info(
                    'Accuracy of the network on the {0} train images: {1:.3f}%'
                    .format(total, accuracy))
                self.logger.info(
                    'Precision of the network on the {0} train images: {1:.3f}'
                    .format(total, precision))
                self.logger.info(
                    'Recall of the network on the {0} train images: {1:.3f}'
                    .format(total, recall))
                self.logger.info(
                    'F1 score of the network on the {0} train images: {1:.3f}'
                    .format(total, f1))

                # Write to Tensorboard file
                self.writer.add_scalar(
                    'Loss/train', self.train_recorder.loss, ep)
                self.writer.add_scalar('Accuracy/train', accuracy, ep)
                self.writer.add_scalar('Precision/train', precision, ep)
                self.writer.add_scalar('Recall/train', recall, ep)
                self.writer.add_scalar('F1/train', f1, ep)

    def test_one_epoch(self, ep):
        self.model.eval()

        self.test_recorder.reset()

        for _, (images, labels) in enumerate(self.validloader, 0):
            images = images.to(self.device)
            labels = labels.to(self.device)

            preds = self.model(images)
            loss = self.criterion(preds, labels)

            self.test_recorder.update(preds, labels, loss.item())

        fp = self.test_recorder.false_positive_counts
        fn = self.test_recorder.false_negative_counts
        tp = self.test_recorder.true_positive_counts
        tn = self.test_recorder.true_negative_counts
        correct = self.test_recorder.correct_counts
        total = self.test_recorder.total_counts

        # Calculate the accuracy, precision, recall, and F1 score
        accuracy = 100.0 * correct / total
        precision = 100.0 * tp / (tp + fp)
        recall = 100.0 * tp / (tp + fn)
        f1 = 100 * (2 * precision * recall / (precision + recall))

        self.logger.info(
            'Loss of the network on the {0} val images: {1:.6f}'.format(
                self.test_recorder.total, self.test_recorder.loss))
        self.logger.info(
            'Accuracy of the network on the {0} val images: {1:.3f}%'.format(
                self.test_recorder.total, accuracy))
        self.logger.info(
            'Precision of the network on the {0} val images: {1:.3f}%'.format(
                self.test_recorder.total, precision))
        self.logger.info(
            'Recall of the network on the {0} val images: {1:.3f}%'.format(
                self.test_recorder.total, recall))
        self.logger.info(
            'F1 of the network on the {0} val images: {1:.3f}%'.format(
                self.test_recorder.total, f1))

        # Compare accuracy
        self.is_best = accuracy > self.best_acc
        if self.is_best:
            self.best_acc = accuracy
            self.best_model = copy.deepcopy(self.model)
            self.best_optim = copy.deepcopy(self.optimizer)
            self.best_epoch = ep

        # Write to Tensorboard file
        self.writer.add_scalar('Loss/val', self.test_recorder.loss, ep)
        self.writer.add_scalar('Accuracy/val', accuracy, ep)
        self.writer.add_scalar('Precision/val', precision, ep)
        self.writer.add_scalar('Recall/val', recall, ep)
        self.writer.add_scalar('F1/val', f1, ep)

def main(): 
    HyphalSolver()