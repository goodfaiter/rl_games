class RewardPlateauStopper:
    """Stops training once the mean episode reward stops improving.

    Config (``params.config.early_stopping``)::

        early_stopping:
          patience: 15      # epochs without improvement before stopping
          min_delta: 0.01   # improvement threshold, relative to |best reward|
          min_epochs: 30    # never stop before this epoch

    An epoch improves on the best reward if it beats it by more than ``min_delta * |best|``.
    """

    def __init__(self, config):
        self.patience = int(config.get('patience', 15))
        self.min_delta = float(config.get('min_delta', 0.01))
        self.min_epochs = int(config.get('min_epochs', 30))
        if self.patience < 1 or self.min_delta < 0.0 or self.min_epochs < 0:
            raise ValueError(f'early_stopping needs patience >= 1, min_delta >= 0, min_epochs >= 0, got {config}')
        self.best = None
        self.best_epoch = 0

    def update(self, epoch_num, mean_reward):
        """Record this epoch's mean reward; returns True when training should stop."""
        if self.best is None or mean_reward > self.best + self.min_delta * abs(self.best):
            self.best = mean_reward
            self.best_epoch = epoch_num
        return epoch_num >= self.min_epochs and epoch_num - self.best_epoch >= self.patience
