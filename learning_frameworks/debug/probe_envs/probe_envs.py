import random
from typing import SupportsFloat, Any, Optional

from gymnasium import Env
from gymnasium.core import ActType, ObsType


class OneRewardProbeTerminal(Env):
    def __init__(self):
        self.timeout_steps = 1

        self.current = 0

    def step(
            self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        self.current += 1
        terminal = self.current >= self.timeout_steps

        return [0], 1, terminal, False, {}

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict[str, Any]] = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        self.current = 0

        return [0], {}


class TwoObsRewardProbeTerminal(Env):
    def __init__(self):
        self.timeout_steps = 1

        self.current = 0
        self.picked_result = 0

    def step(
            self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        self.current += 1
        terminal = self.current >= self.timeout_steps

        return [self.picked_result], self.picked_result, terminal, False, {}

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict[str, Any]] = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        self.current = 0

        self.picked_result = -1 if random.uniform(0, 1) > 0.5 else 1

        return [self.picked_result], {}


class OneThenZeroProbe(Env):
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