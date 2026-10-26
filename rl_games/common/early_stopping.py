class RewardPlateauStopper:
    """Stops training once the mean episode reward stops improving.

    Config (``params.config.early_stopping``)::

        early_stopping:
          patience: 15      # epochs without improvement before stopping
          min_delta: 0.01   # improvement threshold, relative to |best reward|
          min_epochs: 30    # warm-up epochs, not counted for the plateau

    An epoch improves on the best reward if it beats it by more than ``min_delta * |best|``. The warm-up counts
    from the first update (the start of training, or of the last curriculum level, where the agent restarts the
    stopper), so training runs at least ``min_epochs + patience`` epochs from there.
    """

    def __init__(self, config):
        self.patience = int(config.get('patience', 15))
        self.min_delta = float(config.get('min_delta', 0.01))
        self.min_epochs = int(config.get('min_epochs', 30))
        if self.patience < 1 or self.min_delta < 0.0 or self.min_epochs < 0:
            raise ValueError(f'early_stopping needs patience >= 1, min_delta >= 0, min_epochs >= 0, got {config}')
        self.start_epoch = None
        self.best = None
        self.best_epoch = 0

    def update(self, epoch_num, mean_reward):
        """Record this epoch's mean reward; returns True when training should stop."""
        if self.start_epoch is None:
            self.start_epoch = epoch_num
        if epoch_num - self.start_epoch < self.min_epochs:
            return False  # warm-up: the first rewards are not representative
        if self.best is None or mean_reward > self.best + self.min_delta * abs(self.best):
            self.best = mean_reward
            self.best_epoch = epoch_num
        return epoch_num - self.best_epoch >= self.patience

    def epochs_left(self, epoch_num):
        """Epochs until training would stop without an improvement (remaining warm-up + patience)."""
        if self.start_epoch is None:
            return self.min_epochs + self.patience
        warmup_end = self.start_epoch + self.min_epochs
        if self.best is None:
            return max(warmup_end - epoch_num, 0) + self.patience
        return max(self.patience - (epoch_num - self.best_epoch), 0)
