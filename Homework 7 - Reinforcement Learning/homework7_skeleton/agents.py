import random

student_name = "Jonathon Michael Delemos"


# 1. Q-Learning
class QLearningAgent:
    """Implement Q Reinforcement Learning Agent using Q-table."""

    def __init__(self, game, discount, learning_rate, explore_prob):
        """Store any needed parameters into the agent object.
        Initialize Q-table.
        """
        self.game = game
        self.discount = discount
        self.learning_rate = learning_rate
        self.explore_prob = explore_prob
        self.q_table = {}

    def get_q_value(self, state, action):
        """Retrieve Q-value from Q-table.
        For an never seen (s,a) pair, the Q-value is by default 0.
        """
        # initialize unseen (state, action) pairs to 0
        if (state, action) not in self.q_table:
            self.q_table[(state, action)] = 0
            return self.q_table[(state, action)]
        return self.q_table[(state, action)]

    def get_value(self, state):
        """Compute state value from Q-values using Bellman Equation."""
        actions = list(self.game.get_actions(state))
        if not actions:
            return 0.0
        q_values = [self.get_q_value(state, a) for a in actions]
        return max(q_values)

    def get_best_policy(self, state):
        """Compute the best action to take in the state using Policy
        Extraction.
        π(s) = argmax_a Q(s,a)

        If there are ties, return a random one for better performance.
        Hint: use random.choice().
        """
        actions = list(self.game.get_actions(state))
        best_actions = []
        max_q_value = float('-inf')
        for action in actions:
            q_value = self.get_q_value(state, action)
            if q_value > max_q_value:
                max_q_value = q_value
                best_actions = [action]
            elif q_value == max_q_value:
                best_actions.append(action)
        return random.choice(best_actions) if best_actions else None

    def update(self, state, action, next_state, reward):
        """Update Q-values using running average.
        Q(s,a) = (1 - α) Q(s,a) + α (R + γ V(s'))
        Where α is the learning rate, and γ is the discount.

        Note: You should not call this function in your code.
        """
        current_q_value = self.get_q_value(state, action)
        next_value = self.get_value(next_state)
        updated_q_value = (1 - self.learning_rate) * current_q_value + \
            self.learning_rate * (reward + self.discount * next_value)
        self.q_table[(state, action)] = updated_q_value

    # 2. Epsilon Greedy
    def get_action(self, state):
        """Compute the action to take for the agent, incorporating exploration.
        That is, with probability ε, act randomly.
        Otherwise, act according to the best policy.

        Hint: use random.random() < ε to check if exploration is needed.
        """
        actions = list(self.game.get_actions(state))
        for action in actions:
            # here, we decide whether to explore or exploit
            if random.random() < self.explore_prob:
                return random.choice(actions)
            else:
                return self.get_best_policy(state)


# 3. Bridge Crossing Revisited
# Q(s,a) = (1 - α) Q(s,a) + α (R + γ V(s'))
#  updated_q_value = (1 - self.learning_rate) * current_q_value
# + self.learning_rate
# * (reward + self.discount * next_value)
# test / empirical results show that with epsilon = 0.00 and learning rate
# = .2, the agent performs optimally on the bridge crossing problem.
def question3():
    epsilon = .8
    learning_rate = .3
    return "NOT POSSIBLE"
    # If not possible, return 'NOT POSSIBLE'


# 5. Approximate Q-Learning
class ApproximateQAgent(QLearningAgent):
    """Implement Approximate Q Learning Agent using weights."""

    def __init__(self, *args, extractor):
        """Initialize parameters and store the feature extractor.
        Initialize weights table. dict"""
        self.feat_extractor = extractor
        super().__init__(*args)
        self.weights = {}

    def get_weight(self, feature):
        """Get weight of a feature.
        Never seen feature should have a weight of 0.
        """
        if feature not in self.weights:
            self.weights[feature] = 0
        return self.weights[feature]

    def get_q_value(self, state, action):
        """Compute Q value based on the dot product of feature
        Q(s,a) = w_1 * f_1(s,a) + w_2 * f_2(s,a) + ... + w_n * f_n(s,a)
        """
        features = self.feat_extractor(state, action)
        q_value = 0
        for feature, value in features.items():
            if value is None:
                continue
            weight = self.get_weight(feature)
            q_value += weight * value
        return q_value

    def update(self, state, action, next_state, reward):
        """Update weights using least-squares approximation.
        Δ = R + γ V(s') - Q(s,a)
        Then update weights: w_i = w_i + α * Δ * f_i(s, a)
        """
        features = self.feat_extractor(state, action)
        current_q_value = self.get_q_value(state, action)
        next_value = self.get_value(next_state)
        delta = reward + self.discount * next_value - current_q_value
        for feature, value in features.items():
            if value is None:
                continue
            weight = self.get_weight(feature)
            updated_weight = weight + self.learning_rate * delta * value
            self.weights[feature] = updated_weight


# 6. Feedback
# Just an approximation is fine.
feedback_question_1 = """
This assignment took me approximately 6 hours to complete.
Because this was mostly based off math and less coding
logic, I found it easier to follow along with the equations
provided.
"""

feedback_question_2 = """
The hardest part was imagining how the Bellman Equation
propagated over a featureless environment and
how to implement the Q-value updates correctly.
This assignment helped me understand the importance of
 exploration vs exploitation trade-off in reinforcement
 learning.
I learned how to implement function
 approximation using feature extractors.
Overall, I feel better about my understanding
of Q-learning and its applications.

"""

feedback_question_3 = """"
I loved seeing the pacman agent learn to navigate the
environment
using Q-learning techniques. Penn always has
the most interesting GUI demonstrations, it's really
engaging to see the agent improve over time.
"""
