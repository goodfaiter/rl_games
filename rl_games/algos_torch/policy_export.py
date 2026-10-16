"""Export a trained policy as a self-contained TorchScript deployment file.

The exported module owns the observation history, so a deployment loop only feeds the
current (single-frame) observation and gets the action back::

    policy = torch.jit.load("policy.pt", map_location="cpu")
    policy.reset()
    while running:
        action = policy(obs)  # obs: (obs_dim,) -> action: (action_dim,)
        ...  # sleep to keep 1 / policy.frequency

Configured by the optional ``export`` section of ``params.config``::

    export:
      num_hist: 40       # frames stacked into the network input (default 1)
      hist_stride: 10    # steps between stacked frames (default 1)
      frequency: 200.0   # control frequency [Hz] (default 0.0 = unknown)
      action_lower: [..] # per-action command bounds: the policy action a in [-1, 1] is mapped to
      action_upper: [..] # 0.5 * (a + 1) * (upper - lower) + lower (default [-1, 1] = unscaled)
      ...                # any other keys are stored in metadata.json

The env can provide the same keys through ``env_info['export']`` (yaml keys take precedence).
After saving, the file is reloaded and checked against the live model (see ``check_exported_policy``).

Metadata is available both as module attributes (``policy.frequency``, ``policy.num_hist``,
...) and as ``metadata.json`` inside the archive (``torch.jit.load(path, _extra_files=...)``).
"""

import copy
import json
from pathlib import Path

import torch


class ModelWrapper(torch.nn.Module):
    '''
    Main idea is to ignore outputs which we don't need from model
    '''
    def __init__(self, model):
        torch.nn.Module.__init__(self)
        self._model = model

    def forward(self, input_dict):
        input_dict['obs'] = self._model.norm_obs(input_dict['obs'])
        '''
        just model export doesn't work. Looks like onnx issue with torch distributions
        thats why we are exporting only neural network
        '''
        mu, _, _, _ = self._model.a2c_network(input_dict)
        return mu


def trace_policy(model, obs_dim, device='cuda'):
    """Trace ``ModelWrapper(model)`` (obs normalization + actor mean).

    The traced module takes an observation of shape ``(batch, obs_dim)`` and returns ``(mu,)``.
    ``model`` must already be on ``device``.
    """
    from rl_games.algos_torch import flatten

    test_input = {
        'obs': torch.ones((1, obs_dim), device=device, dtype=torch.float32),
        'is_train': False,
        'prev_actions': None,
    }
    with torch.no_grad():
        adapter = flatten.TracingAdapter(ModelWrapper(model), test_input, allow_non_tensor=True)
        return torch.jit.trace(adapter, adapter.flattened_inputs, check_trace=False)


class HistoryPolicy(torch.nn.Module):
    """Stateful policy that stacks single-frame observations into the network input.

    History layout: newest frame first, then every ``hist_stride``-th older frame,
    ``num_hist`` frames in total (as ``TimeSeriesBuffer.get()``). ``reset()`` zeros the
    history, as on an episode reset. The output is the scaled command (see ``scale_actions``).
    """

    frequency: float
    dt: float
    obs_dim: int
    action_dim: int
    num_hist: int
    hist_stride: int
    clip_actions: bool

    def __init__(self, policy, obs_dim, action_dim, num_hist, hist_stride, frequency, clip_actions, action_lower, action_upper):
        super().__init__()
        self.policy = policy  # trace_policy: (1, num_hist * obs_dim) -> (mu,)
        self.frequency = float(frequency)
        self.dt = 1.0 / self.frequency if self.frequency > 0.0 else 0.0
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.num_hist = int(num_hist)
        self.hist_stride = int(hist_stride)
        self.clip_actions = bool(clip_actions)
        self.register_buffer("history", torch.zeros(1 + (self.num_hist - 1) * self.hist_stride, self.obs_dim))
        self.register_buffer("action_lower", torch.as_tensor(action_lower, dtype=torch.float32).reshape(self.action_dim))
        self.register_buffer("action_upper", torch.as_tensor(action_upper, dtype=torch.float32).reshape(self.action_dim))

    @torch.jit.export
    def reset(self) -> None:
        self.history.zero_()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs = obs.reshape(1, self.obs_dim).to(device=self.history.device, dtype=self.history.dtype)
        self.history.copy_(torch.cat([obs, self.history[:-1]], dim=0))
        stacked = self.history[:: self.hist_stride].reshape(1, self.num_hist * self.obs_dim)
        action = self.policy(stacked)[0].reshape(self.action_dim)
        return scale_actions(action, self.action_lower, self.action_upper, self.clip_actions)


def scale_actions(action: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor, clip_actions: bool) -> torch.Tensor:
    """Map a policy action in [-1, 1] to the command range (vlearn ``scale``)."""
    if clip_actions:
        action = torch.clamp(action, -1.0, 1.0)
    return 0.5 * (action + 1.0) * (upper - lower) + lower


def export_history_policy(model, stacked_obs_dim, path, export_config=None, clip_actions=True):
    """Trace ``model`` (an rl_games A2C model) on CPU, save it wrapped in ``HistoryPolicy`` and check it."""
    export_config = dict(export_config or {})
    num_hist = int(export_config.get('num_hist', 1))
    hist_stride = int(export_config.get('hist_stride', 1))
    frequency = float(export_config.get('frequency', 0.0))
    if stacked_obs_dim % num_hist != 0:
        raise ValueError(f"Observation size {stacked_obs_dim} is not divisible by num_hist={num_hist}")
    obs_dim = stacked_obs_dim // num_hist

    model = getattr(model, '_orig_mod', model)  # unwrap torch.compile
    # Frozen CPU copy: the adapter's no_grad is not part of the trace.
    frozen = copy.deepcopy(model).cpu().float().eval().requires_grad_(False)
    traced = trace_policy(frozen, stacked_obs_dim, device='cpu')
    with torch.no_grad():
        action_dim = traced(torch.zeros(1, stacked_obs_dim))[0].shape[-1]
    action_lower = [float(x) for x in export_config.get('action_lower', [-1.0] * action_dim)]
    action_upper = [float(x) for x in export_config.get('action_upper', [1.0] * action_dim)]
    if len(action_lower) != action_dim or len(action_upper) != action_dim:
        raise ValueError(f"action_lower/action_upper must have {action_dim} entries, got {len(action_lower)}/{len(action_upper)}")

    policy = torch.jit.script(
        HistoryPolicy(traced, obs_dim, action_dim, num_hist, hist_stride, frequency, clip_actions, action_lower, action_upper)
    )
    metadata = {
        **export_config,
        'frequency': policy.frequency,
        'dt': policy.dt,
        'num_hist': num_hist,
        'hist_stride': hist_stride,
        'obs_dim': obs_dim,
        'action_dim': action_dim,
        'clip_actions': bool(clip_actions),
        'action_lower': action_lower,
        'action_upper': action_upper,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    policy.save(str(path), _extra_files={'metadata.json': json.dumps(metadata, indent=2)})
    print(f"Saved deployment policy to {path}: {metadata}")
    check_exported_policy(path, model, stacked_obs_dim)
    return path


def check_exported_policy(path, model, stacked_obs_dim, atol=1e-4):
    """Sanity check a saved policy against the live model on an all-ones observation.

    The saved file is fed single-frame observations of 1.0 until its history is full, and its
    action is compared with the live model's scaled action (full model forward on its own device,
    then ``scale_actions``) for a stacked observation of 1.0. Raises ``RuntimeError`` on mismatch.
    """
    policy = torch.jit.load(str(path), map_location='cpu')
    policy.reset()
    obs = torch.ones(policy.obs_dim)
    for _ in range(policy.history.shape[0]):
        exported = policy(obs)

    model = getattr(model, '_orig_mod', model)
    was_training = model.training
    model.eval()  # do not update the running obs statistics
    try:
        device = next(model.parameters()).device
        input_dict = {'obs': torch.ones(1, stacked_obs_dim, device=device), 'is_train': False, 'prev_actions': None}
        with torch.no_grad():
            mu = model(input_dict)['mus'].float().reshape(-1).cpu()
    finally:
        model.train(was_training)
    expected = scale_actions(mu, policy.action_lower, policy.action_upper, policy.clip_actions)

    max_err = (exported - expected).abs().max().item()
    if not torch.allclose(exported, expected, atol=atol, rtol=0.0):
        raise RuntimeError(
            f"Exported policy {path} does not match the live model on an all-ones observation "
            f"(max abs error {max_err:.3g}): exported {exported.tolist()}, expected {expected.tolist()}"
        )
    print(f"Sanity check passed for {path}: action {exported.tolist()} (max abs error {max_err:.3g})")
