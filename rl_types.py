from typing import TypeVar, Tuple, Sequence, Callable

State = TypeVar('State', int, float)
Action = TypeVar('Action', int, float)

ActionDist = Sequence[float]

# Deterministic policy
DetPolicy = Callable[[State], Action]

Reward = TypeVar('Reward', int, float)

# Assuming reward is a function of state only
RewardFunc = Callable[[State], Reward]