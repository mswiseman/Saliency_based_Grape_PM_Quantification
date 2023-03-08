import torch

class Recorder:
    def __init__(self, name):
        """
            The class to record loss history in the training process 
        """
        self._loss = 0
        self.correct_counts = 0
        self.true_positive_counts = 0
        self.false_positive_counts = 0
        self.true_negative_counts = 0
        self.false_negative_counts = 0
        self.total_counts = 0
        self._name = name

    def update(self, preds, labels, loss):
        self.correct_counts += self.count_correct(preds, labels)
        self.false_positive_counts += self.count_false_positives(preds, labels)
        self.false_negative_counts += self.count_false_negatives(preds, labels)
        self.true_positive_counts += self.count_true_positives(preds, labels)
        self.true_negative_counts += self.count_true_negatives(preds, labels)
        self.total_counts += len(labels)
        self._loss += loss

    @property
    def name(self):
        return self._name

    @property
    def loss(self):
        return self._loss
    
    @property
    def correct(self):
        return self.correct_counts
    
    @property
    def total(self):
        return self.total_counts
    
    def reset(self):
        self._loss = 0
        self.correct_counts = 0
        self.true_positive_counts = 0
        self.false_positive_counts = 0
        self.true_negative_counts = 0
        self.false_negative_counts = 0
        self.total_counts = 0

    def count_correct(self, out, yb):
        preds = torch.argmax(out, dim=1)
        return (preds == yb).sum().item()

    def count_false_positives(self, out, yb):
        preds = torch.argmax(out, dim=1)
        return ((preds == 1) & (yb == 0)).sum().item()

    def count_false_negatives(self, out, yb):
        preds = torch.argmax(out, dim=1)
        return ((preds == 0) & (yb == 1)).sum().item()

    def count_true_positives(self, out, yb):
        preds = torch.argmax(out, dim=1)
        return ((preds == 1) & (yb == 1)).sum().item()

    def count_true_negatives(self, out, yb):
        preds = torch.argmax(out, dim=1)
        return ((preds == 0) & (yb == 0)).sum().item()
