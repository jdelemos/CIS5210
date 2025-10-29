# Include your imports here, if any are used.

from collections import defaultdict

student_name = "Jonathon Delemos"


class ValueIterationAgent:
    """Implement Value Iteration Agent using Bellman Equations."""

    def __init__(self, game, discount):
        self.game = game
        self.discount = discount
        self.values = defaultdict(float)
        self.policy = {s: None for s in self.game.states}

    def get_value(self, state):
        """Return value V*(s)."""
        return self.values[state]

    def get_q_value(self, state, action):
        """Return Q*(s,a) using the Bellman equation."""
        if not self.game.get_actions(state):
            return 0
        q = 0.0
        for next_state, prob in self.game.get_transitions(
                state, action).items():
            reward = self.game.get_reward(state, action, next_state)
            q += prob * (reward + self.discount * self.get_value(next_state))
        return q

    def get_best_policy(self, state):
        """Return π*(s) = argmax_a Q*(s,a)."""
        actions = self.game.get_actions(state)
        if not actions:
            return None
        best_action = max(actions, key=lambda a: self.get_q_value(state, a))
        return best_action

    def iterate(self):
        """Run one iteration of value update."""
        new_values = {}
        for state in self.game.states:
            actions = self.game.get_actions(state)
            if not actions:
                new_values[state] = 0
                continue
            new_values[state] = max(self.get_q_value(state, a)
                                    for a in actions)
        self.values.update(new_values)


# 2. Policy Iteration
class PolicyIterationAgent(ValueIterationAgent):
    """Implement Policy Iteration Agent.

    The only difference between policy iteration and value iteration is at
    their iteration method. However, if you need to implement helper function
    or override ValueIterationAgent's methods, you can add them as well.
    """

    def iterate(self):
        """Run single policy iteration.
        Fix current policy, iterate state values V(s) until
        |V_{k+1}(s) - V_k(s)| < ε

        """
        epsilon = 1e-6
        find_policy = True
        while find_policy:
            while True:
                delta = 0.0
                for s in self.game.states:
                    previous_bst_value = self.get_value(s)
                    actions = self.policy[s]
                    if actions is None:
                        continue
                    new_bst_value = self.get_q_value(s, self.policy[s])
                    self.values[s] = new_bst_value
                    delta = max(delta, abs(previous_bst_value - new_bst_value))
                if delta < epsilon:
                    break

            find_policy = False
            for s in self.game.states:
                old_action = self.policy[s]
                actions = self.game.get_actions(s)
                if not actions:
                    continue
                best_action = max(
                    actions,
                    key=lambda a: self.get_q_value(
                        s,
                        a))
                self.policy[s] = best_action
                if old_action != best_action:
                    find_policy = True

# 3. Bridge Crossing Analysis

# Decrease noise: lowers the chance of slipping into −R chasm states,
# monotonically increasing the expected value of the bridge path.

# Or increase discount: makes the distant high terminal more valuable
# relative to the immediate safe one, but this still leaves you with a
# sizable 20% slip risk, so you often need γ very close to 1 to flip the
# policy.


def question_3():
    discount = 0.9
    noise = 0.0
    return discount, noise


# 4. Policies
def question_4a():
    discount = 0.2
    noise = 0.0
    living_reward = -1.0
    return discount, noise, living_reward
    # If not possible, return 'NOT POSSIBLE'


def question_4b():
    discount1 = 0.2
    noise1 = 0.2
    living_reward1 = -0.5
    return discount1, noise1, living_reward1
    # If not possible, return 'NOT POSSIBLE'


def question_4c():
    discount2 = 0.9
    noise2 = 0.0
    living_reward2 = -0.2
    return discount2, noise2, living_reward2
    # If not possible, return 'NOT POSSIBLE'


def question_4d():
    discount = 0.9
    noise = 0.2
    living_reward = -0.05
    return discount, noise, living_reward
    # If not possible, return 'NOT POSSIBLE'


def question_4e():
    discount = 1.0
    noise = 0.0
    living_reward = 1.0
    return discount, noise, living_reward
    # If not possible, return 'NOT POSSIBLE'


# 5. Feedback
# Just an approximation is fine.
feedback_question_1 = """
This assignment took me at least seven hours to complete.
I found the visualization of the bellman equations the
most satisfying
part of the assignment.
The most challenging part was understanding the policy
iteration algorithm and ensuring that
it was implemented correctly.
"""

feedback_question_2 = """
I found the lectures and the textbook to be very helpful in understanding
the concepts needed for this assignment. However, once I began coding,
I had to refer to the structure of the assignment quite often to ensure
I was
implementing the algorithms correctly.
"""

feedback_question_3 = """
I really enjoyed watching the algorithms produce the optimal policies
for the different MDP scenarios. It was rewarding to
see the theoretical concepts
come to life through the code.
"""


