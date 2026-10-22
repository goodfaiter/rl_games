from rl_games.common.early_stopping import RewardPlateauStopper


class RewardPlateauCurriculum:
    """Steps a curriculum factor through ``levels`` each time the mean episode reward plateaus.

    Config (``params.config.curriculum``)::

        curriculum:
          levels: [0.0, 0.25, 0.5, 0.75, 1.0]  # training starts at the first level
          patience: 10      # epochs without improvement that count as a plateau
          min_delta: 0.01   # improvement threshold, relative to |best reward|
          min_epochs: 10    # warm-up epochs of each level, not counted for its plateau

    The plateau rule is the early stopper's, restarted on every level after its first ``min_epochs`` epochs
    (the first rewards of a run, or of a fresh level, are not representative). The agent exposes the current
    ``factor`` as ``curriculum_factor`` and hands it to the env through ``set_train_info``; early stopping
    only applies on the last level.
    """

    def __init__(self, config):
        self.levels = [float(level) for level in config.get('levels', [])]
        if not self.levels:
            raise ValueError(f'curriculum needs a non-empty list of levels, got {config}')
        self.plateau_config = {key: config[key] for key in ('patience', 'min_delta') if key in config}
        self.plateau_config['min_epochs'] = 0
        self.warmup_epochs = int(config.get('min_epochs', 10))
        self.level = 0
        self._start_level(0)

    def _start_level(self, epoch_num):
        self.level_start_epoch = epoch_num
        self.stopper = RewardPlateauStopper(self.plateau_config)

    @property
    def factor(self):
        return self.levels[self.level]

    @property
    def finished(self):
        return self.level == len(self.levels) - 1

    def update(self, epoch_num, mean_reward):
        """Record this epoch's mean reward; returns True when the curriculum moved to the next level."""
        level_epoch = epoch_num - self.level_start_epoch
        if self.finished or level_epoch < self.warmup_epochs or not self.stopper.update(level_epoch, mean_reward):
            return False
        self.level += 1
        self._start_level(epoch_num)
        return True

