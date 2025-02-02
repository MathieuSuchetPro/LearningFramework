"""
Environments implementations from https://andyljones.com/posts/rl-debugging.html#probe
"""

import random
from typing import SupportsFloat, Any, Optional

from gymnasium import Env
from gymnasium.core import ActType, ObsType


class OneRewardProbeTerminal(Env):
    """
    Probe environment that only returns +1 reward and lasts 1 step, used to check if the value

    From https://andyljones.com/posts/rl-debugging.html#probe
    This isolates the value network. If my agent can't learn that the value of the only observation it ever sees it 1, there's a problem with the value loss calculation or the optimizer.
    """

    def step(
            self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        return [0], 1, True, False, {}

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict[str, Any]] = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        return [0], {}


class TwoObsRewardProbeTerminal(Env):
    """
    Probe environment that returns the -1 or 1 based on a random number
    Observation and reward are linked

    From https://andyljones.com/posts/rl-debugging.html#probe
    If my agent can learn the value in (1.) but not this one - meaning it can learn a constant reward but not a predictable one! - it must be that backpropagation through my network is broken.
    """
    def __init__(self):
        self.picked_result = 0

    def step(
            self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        return [self.picked_result], self.picked_result, True, False, {}

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict[str, Any]] = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        self.picked_result = -1 if random.uniform(0, 1) > 0.5 else 1
        return [self.picked_result], {}


class OneThenZeroProbe(Env):
    """
    Probe environment that outputs 0 then 1 for reward and obs

    From https://andyljones.com/posts/rl-debugging.html#probe
    If my agent can learn the value in (2.) but not this one, it must be that my reward discounting is broken.
    """

    def __init__(self):
        self.timeout_steps = 2

        self.current = 0
        self.picked_result = 0

    def step(
            self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        self.current += 1

        terminal = self.current >= self.timeout_steps

        return [1 if terminal else 0], 1 if terminal else 0, terminal, False, {}

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict[str, Any]] = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        self.current = 0

        return [0], {}


class TwoActionProbe(Env):
    """
    Probe environment with 2 actions, actions are linked with the reward

    From https://andyljones.com/posts/rl-debugging.html#probe
    The first env to exercise the policy!
    If my agent can't learn to pick the better action, there's something wrong with either my advantage calculations, my policy loss or my policy update.
    That's three things, but it's easy to work out by hand the expected values for each one and check that the values produced by your actual code line up with them.
    """
    def step(
            self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        action_map = {
            0: -1,
            1: 1
        }

        return [0], action_map[int(action)], True, False, {}

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict[str, Any]] = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        return [0], {}


class TwoActionTwoObsDependantProbe(Env):
    """
    Probe environment that has 2 actions, 2 observations, 2 rewards, all linked

    From https://andyljones.com/posts/rl-debugging.html#probe
    Now we've got a dependence on both obs and action.
    The policy and value networks interact here, so there's a couple of things to verify:
        - that the policy network learns to pick the right action in each of the two states
        - that the value network learns that the value of each state is +1.
    If everything's worked up until now, then if - for example - the value network fails to learn here, it likely means your batching process is feeding the value network stale experience.
    """
    def __init__(self):
        self.timeout_steps = 1

        self.current = 0
        self.picked_result = 0

    def step(
            self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        self.current += 1
        terminal = self.current >= self.timeout_steps

        action_map = {
            0: -1,
            1: 1
        }

        result = min(self.picked_result, action_map[int(action)])

        return [self.picked_result], result, terminal, False, {}

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict[str, Any]] = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        self.current = 0

        self.picked_result = -1 if random.uniform(0, 1) > 0.5 else 1

        return [self.picked_result], {}


class MultiDiscreteEnv(Env):
    """
    Test environment for multi discrete policies
    """
    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:

        result = action.sum()

        return [0], result, True, False, {}

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict[str, Any]] = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        return [0], {}