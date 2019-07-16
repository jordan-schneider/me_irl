import classic_irl
import unittest
import gym


class TestClassicIrlMethods(unittest.TestCase):
    def setUp(self):
        self.env = gym.make("FrozenLake-v0")
        self.n_states = self.env.observation_space.n
        self.n_actions = self.env.action_space.n

    def test_rename_actions(self):
        P = self.env.P
        go_left = lambda x: 0
        go_down = lambda x: 1
        go_right = lambda x: 2
        go_up = lambda x: 3
        self.assertEqual(P, classic_irl.rename_actions(P, go_left))

        P_down = classic_irl.rename_actions(P, go_down)
        for state in range(self.n_states):
            self.assertEqual(P[state][0], P_down[state][1])
            self.assertEqual(P[state][1], P_down[state][0])
            self.assertEqual(P[state][2], P_down[state][2])
            self.assertEqual(P[state][3], P_down[state][3])

        P_right = classic_irl.rename_actions(P, go_right)
        for state in range(self.n_states):
            self.assertEqual(P[state][0], P_right[state][2])
            self.assertEqual(P[state][1], P_right[state][1])
            self.assertEqual(P[state][2], P_right[state][0])
            self.assertEqual(P[state][3], P_right[state][3])

        P_up = classic_irl.rename_actions(P, go_up)
        for state in range(self.n_states):
            self.assertEqual(P[state][0], P_up[state][3])
            self.assertEqual(P[state][1], P_up[state][1])
            self.assertEqual(P[state][2], P_up[state][2])
            self.assertEqual(P[state][3], P_up[state][0])

    def test_P_to_array(self):
        P_array = classic_irl.P_to_array(self.env.P)

        # Test that each row represents a probability distribution
        for action in range(self.n_actions):
            for start_state in range(self.n_states):
                self.assertEqual(P_array[action][start_state].sum(), 1)

    def test_classic_irl(self):
        class StubEnv(gym.Env):
            self.P = {}
            self.observation_space = None
            self.action_space = None

        env = StubEnv()
        # Three states, 0 <=> 1 <=> 2, action 0 is go left, action 1 is stay, action 2 is go right
        # No reward, never done
        env.P = {
            0: {0: [(1, 0, 0, 0)], 1: [(1, 0, 0, 0)], 2: [(1, 1, 0, 0)]},
            1: {0: [(1, 0, 0, 0)], 1: [(1, 1, 0, 0)], 2: [(1, 2, 0, 0)]},
            2: {0: [(1, 1, 0, 0)], 1: [(1, 2, 0, 0)], 2: [(1, 2, 0, 0)]},
        }
        env.observation_space = gym.spaces.Discrete(3)
        env.action_space = gym.spaces.Discrete(3)

        for gamma in [0.9, 0.99, 0.999]:
            for r_max in [1, 10, 100]:
                for lmbda in [0, 1, 1.5, 2]:
                    reward_f = classic_irl.classic_irl(
                        env, lambda x: 0, gamma, r_max, lmbda
                    )
                    self.assertGreaterEqual(reward_f(0), reward_f(1))
                    self.assertGreaterEqual(reward_f(1), reward_f(2))


if __name__ == "__main__":
    unittest.main()
