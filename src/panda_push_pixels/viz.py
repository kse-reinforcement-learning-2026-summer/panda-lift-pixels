"""Rollout + visualization helpers shared by exploration notebooks and training scripts.

Unlike :mod:`panda_push_pixels.grading`, this module is not part of the graded contract — it is
a convenience for *looking at* the environment (random or trained rollouts) without every
notebook hand-rolling the same loop. It never imports Stable-Baselines3 or a specific policy
framework: ``policy`` is either omitted (uniform-random actions), a ``torch`` module taking a
batched observation tensor (a loaded ``model.pt`` or ``extract_actor`` output), or any plain
``obs -> action`` callable (e.g. ``lambda obs: model.predict(obs, deterministic=True)[0]``).
"""

import numpy as np
import torch

from .contract import ACTION_HIGH, ACTION_LOW
from .env import PandaPushPixels


def _resolve_action(policy, obs, env):
    if policy is None:
        return env.action_space.sample()
    if isinstance(policy, torch.nn.Module) or isinstance(policy, torch.jit.ScriptModule):
        with torch.no_grad():
            tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            action = policy(tensor).cpu().numpy().reshape(-1)
        return np.clip(action, ACTION_LOW, ACTION_HIGH)
    return np.clip(np.asarray(policy(obs)), ACTION_LOW, ACTION_HIGH)


def render_episode(env=None, policy=None, seed=0):
    """Roll out one episode and collect the human-viewable frame + trajectory at every step.

    ``env`` defaults to a fresh :class:`PandaPushPixels` (closed automatically); pass your own to
    reuse one (e.g. with bigger ``render_kwargs`` for a nicer preview) — it is left open. ``policy``
    is ``None`` (random actions), a torch module, or a plain ``obs -> action`` callable.

    Returns a dict with ``frames`` (list of ``(H, W, 3)`` uint8 arrays, one per step *including*
    the initial reset), ``infos``, ``actions``, ``rewards``, ``episode_return`` and ``success``.
    """
    own_env = env is None
    if own_env:
        env = PandaPushPixels()

    obs, info = env.reset(seed=seed)
    frames = [env.render()]
    infos = [info]
    actions, rewards = [], []
    episode_return, success, done = 0.0, False, False

    while not done:
        action = _resolve_action(policy, obs, env)
        obs, reward, terminated, truncated, info = env.step(action)
        frames.append(env.render())
        actions.append(action)
        rewards.append(reward)
        infos.append(info)
        episode_return += reward
        success = bool(info["is_success"])
        done = terminated or truncated

    if own_env:
        env.close()

    return {
        "frames": frames,
        "infos": infos,
        "actions": actions,
        "rewards": rewards,
        "episode_return": episode_return,
        "success": success,
    }


def save_video(frames, path, fps=20):
    """Save a list of ``(H, W, 3)`` uint8 frames (e.g. ``render_episode(...)["frames"]``) to
    ``path`` — format (``.mp4``, ``.gif``, ...) is inferred from the extension.

    Requires ``imageio`` (the ``[train]`` extra; ``.mp4`` needs its ``ffmpeg`` plugin) — imported
    lazily so the rest of this module works without it.
    """
    import imageio

    imageio.mimsave(path, frames, fps=fps)
    return path
