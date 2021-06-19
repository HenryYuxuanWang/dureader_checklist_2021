import numpy as np


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_score, model, output_dir):

        if self.best_score is None:
            self.best_score = val_score
            self.save_checkpoint(val_score, model, output_dir)
        elif val_score > self.best_score:
            self.best_score = val_score
            self.save_checkpoint(val_score, model, output_dir)

    def save_checkpoint(self, val_score, model, path):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(u'  val_score: %.5f, best_val_score: %.5f\n' % (val_score, self.best_score))
        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(path)
