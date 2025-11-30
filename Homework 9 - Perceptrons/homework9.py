############################################################
# CIS 521: Homework 9
############################################################

############################################################
# Imports
############################################################

# Include your imports here, if any are used.
import homework9_data as data
from collections import defaultdict

############################################################

student_name = "Jonathon Delemos"

############################################################
# Section 1: Perceptrons
############################################################


class BinaryPerceptron(object):
    def __init__(self, examples, iterations):
        self.iterations = iterations
        self.weights = defaultdict(float)
        self._train(examples)

    def predict(self, x):
        return self._score(x) > 0

    def _train(self, examples):
        for _ in range(self.iterations):
            for features, label in examples:
                target = 1 if label else -1
                score = self._score(features)
                if score * target <= 0:
                    for key, value in features.items():
                        self.weights[key] += target * value

    def _score(self, features):
        return sum(
            self.weights[key] * value for key, value in features.items()
        )


class MulticlassPerceptron(object):

    def __init__(self, examples, iterations):
        self.iterations = iterations
        self.labels = []
        seen = set()
        for _, label in examples:
            if label not in seen:
                seen.add(label)
                self.labels.append(label)
        self.weights = {label: defaultdict(float) for label in self.labels}
        if not self.labels:
            raise ValueError("Training data must include at least one label.")
        self._train(examples)

    def predict(self, x):
        return self._predict_label(x)

    def _train(self, examples):
        for _ in range(self.iterations):
            for features, label in examples:
                predicted_label = self._predict_label(features)
                if predicted_label != label:
                    for key, value in features.items():
                        if value != 0:
                            # “This example should look more like class label
                            # and less like class predicted_label.”
                            self.weights[label][key] += value
                            self.weights[predicted_label][key] -= value

    def _predict_label(self, features):
        best_label = self.labels[0]
        best_score = self._score(best_label, features)
        for label in self.labels[1:]:
            score = self._score(label, features)
            if score > best_score:
                best_score = score
                best_label = label
        return best_label

    def _score(self, label, features):
        return sum(
            self.weights[label][key] * value
            for key, value in features.items()
        )

############################################################
# Section 2: Applications
############################################################


class IrisClassifier(object):

    def __init__(self, data):
        training = [(self._vector_to_features(vector), label)
                    for vector, label in data]
        self.perceptron = MulticlassPerceptron(training, iterations=30)

    def classify(self, instance):
        return self.perceptron.predict(self._vector_to_features(instance))

    def _vector_to_features(self, vector):
        features = {f"x{i+1}": value for i, value in enumerate(vector)}
        features["bias"] = 1.0
        return features


class DigitClassifier(object):

    def __init__(self, data):
        training = [(self._vector_to_features(vector), label)
                    for vector, label in data]
        self.perceptron = MulticlassPerceptron(training, iterations=50)

    def classify(self, instance):
        return self.perceptron.predict(self._vector_to_features(instance))

    def _vector_to_features(self, vector):
        features = {
            f"p{i+1}": value
            for i, value in enumerate(vector)
            if value
        }
        features["bias"] = 1.0
        return features


class BiasClassifier(object):

    def __init__(self, data):
        training = [(self._augment_features(value), label)
                    for value, label in data]
        self.perceptron = BinaryPerceptron(training, iterations=25)

    def classify(self, instance):
        return self.perceptron.predict(self._augment_features(instance))

    def _augment_features(self, value):
        return {"x": value, "bias": 1.0}


class MysteryClassifier1(object):

    def __init__(self, data):
        training = [(self._augment_features(vector), label)
                    for vector, label in data]
        self.perceptron = BinaryPerceptron(training, iterations=50)

    def classify(self, instance):
        return self.perceptron.predict(self._augment_features(instance))

    def _augment_features(self, vector):
        x1, x2 = vector
        radius = x1 * x1 + x2 * x2
        return {
            "x1": x1,
            "x2": x2,
            "x1_sq": x1 * x1,
            "x2_sq": x2 * x2,
            "radius": radius,
            "bias": 1.0,
        }


class MysteryClassifier2(object):

    def __init__(self, data):
        training = [(self._augment_features(vector), label)
                    for vector, label in data]
        self.perceptron = BinaryPerceptron(training, iterations=80)

    def classify(self, instance):
        return self.perceptron.predict(self._augment_features(instance))

    def _augment_features(self, vector):
        x1, x2, x3 = vector
        prod = x1 * x2 * x3
        return {"prod": prod, "bias": 1.0}

############################################################
# Section 3: Feedback
############################################################


# Just an approximation is fine.
feedback_question_1 = """
This assignment took approximately 8 hours to complete.
"""

feedback_question_2 = """
I found the most challenging part
to be fully understanding the multiclass perceptron training process.
Sort of ironically, once I had coded the solution, it made a
lot more sense to me. Usually I understand the theory first,
but in this case implementing it helped solidify my understanding.
"""

feedback_question_3 = """
I liked the structure of the assignment,
which gradually built up from binary to multiclass perceptrons,
and then to applications. It was a great way to learn
about updating weights in perceptrons.
"""


# if __name__ == "__main__":
#     # train = [({"x1": 1}, True), ({"x2": 1}, True),
#     #          ({"x1": -1}, False), ({"x2": -1}, False)]
#     # test = [{"x1": 1}, {"x1": 1, "x2": 1},
#     #         {"x1": -1, "x2": 1.5}, {"x1": -0.5, "x2": -2}]
#     # p = BinaryPerceptron(train, 1)
#     # print([p.predict(x) for x in test])
#     train = [({"x1": 1}, 1), ({"x1": 1, "x2": 1}, 2), ({"x2": 1}, 3),
#     ({"x1": -1, "x2": 1}, 4), ({"x1": -1}, 5), ({"x1": -1, "x2": -1}, 6),
#     ({"x2": -1}, 7), ({"x1": 1, "x2": -1}, 8)]

#     p = MulticlassPerceptron(train, 10)
#     print([p.predict(x) for x,y in train])


# train = [({"x1": 1}, 1), ({"x1": 1, "x2": 1}, 2), ({"x2": 1}, 3),
# ({"x1": -1, "x2": 1}, 4), ({"x1": -1}, 5), ({"x1": -1, "x2": -1}, 6),
# ({"x2": -1}, 7), ({"x1": 1, "x2": -1}, 8)]
# # Train the classifier for 10 iterations so that it can learn each class
# p = MulticlassPerceptron(train, 10)
# # Test whether the classifier correctly learned the training data
# print([p.predict(x) for x, y in train])
