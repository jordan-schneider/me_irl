""" Implementing the classic known-MDP IRL algorithm in Ng, Russell 2000. """

import logging
from typing import Dict, List, Sequence, Tuple

import gym
import numpy as np
import pulp
import copy

from rl_types import DetPolicy, Reward, RewardFunc, State, Action

NestedDicts = Dict[State, Dict[Action, Sequence[Tuple[float, State, Reward, bool]]]]


def rename_actions(P: NestedDicts, policy: DetPolicy) -> NestedDicts:
    """ Renames actions in P so that the policy action is always 0."""
    out: NestedDicts = {}
    for start_state, actions in P.items():
        new_actions = copy.copy(actions)
        policy_action = policy(start_state)
        new_actions[0], new_actions[policy_action] = actions[policy_action], actions[0]
        out[start_state] = new_actions
    return out


def P_to_array(P: NestedDicts) -> np.array:
    """ Converts a transition matrix in nested dictionary format to a numpy array.

    P is usually given as starting state -> action -> ending state w/ data, we reorder this to
    action -> starting state -> ending state -> transition probability.
    """
    # Action, Starting State, Ending State, value is probability
    out = np.zeros(shape=(len(P[0]), len(P), len(P)))
    for start_state, actions in P.items():
        for action, results in actions.items():
            for prob, end_state, _, __ in results:
                out[action, start_state, end_state] += prob
    return out


def lp_min(
    problem: pulp.LpProblem, values: Sequence[pulp.LpAffineExpression], name: str = ""
) -> pulp.LpVariable:
    """ Augments a problem to add a layer of minimization to the objective.

    Returns a proxy LpVariable representing the minimum of the given values.
    """
    min_value = pulp.LpVariable(name=name)
    for value in values:
        # min_value <= value
        # value - min_value >= 0
        problem.addConstraint(value - min_value >= 0)
    return min_value


def classic_irl(
    env: gym.Env, policy: DetPolicy, gamma: float, r_max: float, lmbda: float
) -> RewardFunc:
    """ Returns a reward function that makes the given policy optimal.

    Uses the classic linear programming approach from "Algorithms for Inverse Reinforcement
    Learning" by Ng, Russell. Assumes a finite state and action space, and a deterministic policy.
    Environment must have a transition matrix P in NestedDict format.
    """
    if gamma > 1 or gamma < -1:
        raise ValueError("|gamma| must be less than 1.")
    if r_max <= 0:
        raise ValueError("r_max must be positive.")
    if lmbda < 0:
        raise ValueError("lmbda must be non-negative.")
    if not isinstance(env.observation_space, gym.spaces.Discrete):
        raise ValueError("Observation space must be gym.spaces.Discrete.")
    if not isinstance(env.action_space, gym.spaces.Discrete):
        raise ValueError("Action space must be gym.spaces.Discrete")

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    problem = pulp.LpProblem("classicIRL", pulp.LpMaximize)
    reward_vars = pulp.LpVariable.dicts(
        name="r", indexs=range(n_states), lowBound=-r_max, upBound=r_max
    )

    P = P_to_array(rename_actions(env.P, policy))
    P_policy = P[0]

    right_matrix = np.linalg.inv(np.eye(n_states) - gamma * P_policy)

    for other_action in range(1, n_actions):
        left_matrix = P_policy - P[other_action]
        coefficients = np.matmul(left_matrix, right_matrix)
        for row in coefficients:
            assert len(row) == len(reward_vars)
            problem.addConstraint(
                pulp.lpSum([reward_vars[i] * row[i] for i in reward_vars.keys()]) >= 0
            )

    objective = pulp.LpAffineExpression(name="objective")
    for i in range(len(P[0])):
        alternative_policy_distances: List[pulp.LpAffineExpression] = list()
        for other_action in range(1, n_actions):
            left = P_policy[i] - P[other_action][i]
            coefficients = np.matmul(left, right_matrix)
            assert len(coefficients) == len(reward_vars)
            alternative_policy_distances.append(
                pulp.lpSum(
                    [reward_vars[i] * coefficients[i] for i in reward_vars.keys()]
                )
            )
        min_value = lp_min(problem, alternative_policy_distances, name=f"row_{i}")
        objective += min_value

    # Regularization term to penalize more complex reward functions
    for r_var in reward_vars.values():
        # abs is just -min(x, -x)
        neg_abs_value = lp_min(
            problem, [r_var, -r_var], name=f"neg_abs_{r_var.getName()}"
        )
        # subtracting negative min(r_i, -r_i) is just adding min(r_i, -r_i)
        objective += lmbda * neg_abs_value

    problem.setObjective(objective)
    problem.solve()

    for constraint in problem.constraints.values():
        if not constraint.valid():
            logging.warning(f"constraint: {constraint} violated.")

    logging.info(f"Status is {pulp.LpStatus[problem.status]}")

    return lambda x: reward_vars[x].value()


def check_reward(env: gym.Env, reward_f: RewardFunc, policy: DetPolicy, gamma: float):
    """ Verifies that the reward function makes the policy optimal in the environment.

    Assumes finite state and action spaces, and reward as a function of state.
    """
    if not isinstance(env.observation_space, gym.spaces.Discrete):
        raise ValueError("State space must be gym.spaces.Discrete")
    if not isinstance(env.action_space, gym.spaces.Discrete):
        raise ValueError("Action space must be gym.spaces.Discrete")

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    r = np.array([reward_f(state) for state in range(n_states)])

    P = P_to_array(rename_actions(env.P, policy))
    P_policy = P[0]

    right_matrix = np.matmul(np.linalg.inv(np.eye(n_states) - gamma * P_policy), r.T)
    assert right_matrix.shape == (n_states,)
    for action in range(1, n_actions):
        left_matrix = P_policy - P[action]
        assert left_matrix.shape == (n_states, n_states)
        result = np.matmul(left_matrix, right_matrix.T)
        if any(result < 0):
            logging.warning(
                f"Recovered reward invalid because for action {action},\n P_0 {P_policy},\n P_a {P[action]},\n gamma {gamma},\n r {r},\n right_matrix {right_matrix},\n left_matrix {left_matrix},\n result is {result}\n which has negative component."
            )
            return False

    return True


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    env = gym.make("FrozenLake-v0")
    logging.info(env.desc)

    # The go left policy is usually suboptimal for all possible reward functions, so expect some
    # garbage.
    go_left = lambda x: 0
    REWARD_F = classic_irl(env=env, policy=go_left, gamma=0.99, r_max=1, lmbda=1)

    for state in range(16):
        logging.info(f" r({state})={REWARD_F(state)}")

    logging.info(
        f"Inferred reward valid: {check_reward(env=env, reward_f=REWARD_F, policy=go_left, gamma=0.99)}"
    )
