class RewardPlateau:
    """Signals once the mean episode reward stops improving: no improvement by more than ``min_delta * |best|`` for
    ``patience`` epochs."""

    def __init__(self, patience, min_delta):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        if self.patience < 1 or self.min_delta < 0.0:
            raise ValueError(f'reward plateau needs patience >= 1 and min_delta >= 0, got {patience}, {min_delta}')
        self.best = None
        self.best_epoch = 0

    def update(self, epoch_num, mean_reward):
        """Record this epoch's mean reward; returns True on a plateau."""
        if self.best is None or mean_reward > self.best + self.min_delta * abs(self.best):
            self.best = mean_reward
            self.best_epoch = epoch_num
        return epoch_num - self.best_epoch >= self.patience

    def epochs_left(self, epoch_num):
        """Epochs until a plateau without an improvement."""
        if self.best is None:
            return self.patience
        return max(self.patience - (epoch_num - self.best_epoch), 0)


class RewardPlateauCurriculum:
    """Steps a curriculum factor through ``levels`` each time the mean episode reward plateaus, and stops training
    when the last level plateaus (the early stopping: ``levels: [1.0]`` is plain early stopping).

    Config (``params.config.curriculum``)::

        curriculum:
          levels: [0.0, 0.25, 0.5, 0.75, 1.0]  # training starts at the first level
          patience: 10      # epochs without improvement that count as a plateau
          min_delta: 0.01   # improvement threshold, relative to |best reward|
          min_epochs: 10    # warm-up epochs of each level, not counted for its plateau

    The plateau restarts on every level after its first ``min_epochs`` epochs (the first rewards of a run, or of a
    fresh level, are not representative), so every level runs at least ``min_epochs + patience`` epochs. The agent
    exposes the current ``factor`` as ``curriculum_factor`` and hands it to the env through ``set_train_info``.
    """

    NEXT_LEVEL = 'next_level'
    STOP = 'stop'

    def __init__(self, config):
        self.levels = [float(level) for level in config.get('levels', [])]
        if not self.levels:
            raise ValueError(f'curriculum needs a non-empty list of levels, got {config}')
        self.patience = int(config.get('patience', 10))
        self.min_delta = float(config.get('min_delta', 0.01))
        self.warmup_epochs = int(config.get('min_epochs', 10))
        if self.warmup_epochs < 0:
            raise ValueError(f'curriculum needs min_epochs >= 0, got {config}')
        self.level = 0
        self._start_level(0)

    def _start_level(self, epoch_num):
        self.level_start_epoch = epoch_num
        self.plateau = RewardPlateau(self.patience, self.min_delta)

    @property
    def factor(self):
        return self.levels[self.level]

    @property
    def finished(self):
        return self.level == len(self.levels) - 1

    def update(self, epoch_num, mean_reward):
        """Record this epoch's mean reward. Returns NEXT_LEVEL when the curriculum moved to the next level, STOP when
        the last level plateaued (training is done), else None."""
        level_epoch = epoch_num - self.level_start_epoch
        if level_epoch < self.warmup_epochs or not self.plateau.update(level_epoch, mean_reward):
            return None
        if self.finished:
            return self.STOP
        self.level += 1
        self._start_level(epoch_num)
        return self.NEXT_LEVEL

    def epochs_left(self, epoch_num):
        """Epochs until the next level (or the stop, on the last level) without an improvement."""
        level_epoch = epoch_num - self.level_start_epoch
        if level_epoch < self.warmup_epochs:
            return self.warmup_epochs - level_epoch + self.patience
        return self.plateau.epochs_left(level_epoch)
