import numpy as np
import nltk

############################################################
# These imports are to assist with Section 7.
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger_eng')
############################################################

############################################################
# CIS 521: Homework 1
############################################################

student_name = "Jonathon Michael Delemos"

# This is where your grade report will be sent.
student_email = "delemos1@seas.upenn.edu"

############################################################
# Section 1: Python Concepts
############################################################

python_concepts_question_1 = "Python is a language that is both strong and "
"dynamically typed. Dynamic typed languages allow  "
"for the variable type to be determined "
"at run time - programmers don't "
"explicitly assign type while coding "
"like in Java or C. Strong typing means that "
"variables "
"don't automatically change type when "
"interacted with. This means that in python while some "
"variables might hold string values of "
"numbers, they can be reassigned to integers later in "
"the program without causing an error. "
"The compiler will automatically handle the underlying "
"memory mangagement. This is a huge advantage to "
"programmers who prefer flexibility "
"when working with their variables."

python_concepts_question_2 = "Python dictionary keys must be "
"hashable, and immutable. "
"To correct the example you must swap the key and data positions."

python_concepts_question_3 = "The second function is "
"significantly faster "
"because strings are immutable. We see each loop iteration as a n time "
"moving through a dataset. The second is a integer result, as there "
"is no loops occuring."

############################################################
# Section 2: Working with Lists
############################################################


def extract_and_apply(lst, p, f):
    return [f(x) for x in lst if p(x) is True]


# item for row in matrix, thats the rows, then for item in row is the
# individual
def concatenate(seqs):
    return [y for x in seqs for y in x]


# this was pretty tricky
def transpose(matrix):
    rows = len(matrix)
    columns = len(matrix[0])
    transposed_matrix = [[0 for _ in range(rows)] for _ in range(columns)]
    for i in range(rows):
        for j in range(columns):
            transposed_matrix[j][i] = matrix[i][j]
    return transposed_matrix


############################################################
# Section 3: Sequence Slicing
############################################################


def copy(seq):
    return seq[:]


def all_but_last(seq):
    return seq[0:-1]


def every_other(seq):
    return (seq[::2])


############################################################
# Section 4: Combinatorial Algorithms
############################################################


def prefixes(seq):
    for i in range(len(seq) + 1):
        yield seq[:i]


def suffixes(seq):
    for i in range(len(seq) + 1):
        yield seq[i:]


def slices(seq):
    for i in range(len(seq)):
        for j in range(len(seq) + 1):
            if j > i:
                yield seq[i:j]

############################################################
# Section 5: Text Processing
############################################################


def normalize(text):
    new_text = text.lower().strip().split()
    return ' '.join(new_text)


def no_vowels(text):
    new_text = list(
        x for x in text if x not in [
            'A',
            'a',
            'E',
            'e',
            'I',
            'i',
            'O',
            'o',
            'U',
            'u'])
    return ''.join(new_text)


def digits_to_words(text):
    result = []
    new = list(text)
    for x in range(len(new)):
        if new[x] == '1':
            result.append('one')
        elif new[x] == '2':
            result.append('two')
        elif new[x] == '3':
            result.append('three')
        elif new[x] == '4':
            result.append('four')
        elif new[x] == '5':
            result.append('five')
        elif new[x] == '6':
            result.append('six')
        elif new[x] == '7':
            result.append('seven')
        elif new[x] == '8':
            result.append('eight')
        elif new[x] == '9':
            result.append('nine')
        elif new[x] == '0':
            result.append('zero')
        else:
            # kind of just a dummy place holder
            y = 0
    return ' '.join(result)


def to_mixed_case(name):
    name = name.lower()
    name = list(name)
    for x in range(len(name)):
        if name[x] == '_':
            name[x] = ' '
    name = ''.join(name).strip().split()
    for x in range(len(name)):
        if x > 0:
            s_name = name[x]
            s_name = list(s_name)
            s_name[0] = s_name[0].upper()
            name[x] = ''.join(s_name)
    return (''.join(name))


############################################################
# Section 6: Polynomials
############################################################


class Polynomial(object):

    def __init__(self, polynomial):
        self.polynomial = tuple(polynomial)

    def get_polynomial(self):
        return self.polynomial

    def __neg__(self):
        new_polynomial = []
        for x, y in self.polynomial:
            new_polynomial.append((-x, y))
        return Polynomial(new_polynomial)

    def __add__(self, other):
        return Polynomial(self.polynomial + other.polynomial)

    def __sub__(self, other):
        new_poly = other.__neg__()
        return Polynomial(self.polynomial + new_poly.polynomial)

    def __mul__(self, other):
        new_poly = []
        for x, y in self.polynomial:
            for f, g in other.polynomial:
                new_poly.append((x * f, y + g))
        return Polynomial(tuple(new_poly))

    def __call__(self, x):
        new_poly = []
        for f, g in self.polynomial:
            new_poly.append(f * (x**g))
        return sum(new_poly)

    def simplify(self):
        # declare an empty list
        result = []
        # defend against empty lists
        if len(self.polynomial) == 0:
            # learned today this means a tuple with one term
            self.polynomial = ((0, 0),)
            return
        # meet the immutability requirements
        new_list = list(self.polynomial)
        # sort it descending per the requirements - this is key
        new_list.sort(key=lambda sorting: sorting[1], reverse=True)
        # grab the comparison variables
        exp = new_list[0][1]
        cur = new_list[0][0]
        # loop through with a slice comparing the later variables with the
        # comparison variables
        for x, y in new_list[1:]:
            if y == exp:
                cur += x
            # update comparison variables if they dont match
            else:
                result.append((cur, exp))
                cur = x
                exp = y
        # append the final pair
        result.append((cur, exp))
        # defend against zero pairs
        final_list = []
        for x, y in result:
            if x != 0:
                final_list.append((x, y))
        # if empty, add a pair of zero
        if len(final_list) == 0:
            self.polynomial = ((0, 0),)
            return
        # turn it back into a tuple
        self.polynomial = tuple(final_list)

    def __str__(self):
        result = []
        counter = 0
        new_list = list(self.polynomial)
        for x, y in new_list:
            # determine positive or negative value
            if x < 0:
                sign = '-'
            else:
                sign = '+'
            if x == 0:
                sign = '+'
            # after sign has been gathered, take the abs value, save for
            # formatting
            abs_x = abs(x)
            # this was an absolute mess
            # first term - zero exponent
            if x == 0 and counter == 0 and y == 0 and sign == '-':
                result.append('0')
            elif x == 0 and counter == 0 and y == 0 and sign == '+':
                result.append('0')
            # first term - zero exponent
            elif counter == 0 and y == 0 and sign == '-' and abs_x == 1:
                result.append(str(-1))
            elif counter == 0 and y == 0 and sign == '+' and abs_x == 1:
                result.append(str(1))
            # first term - linear value exponent
            elif counter == 0 and y == 1 and sign == '-' and abs_x == 1:
                result.append(str('-x'))
            elif counter == 0 and y == 1 and sign == '+' and abs_x == 1:
                result.append(str('x'))
            # first term - linear value exponent
            elif counter == 0 and y == 1 and sign == '-':
                result.append(str('-' + str(abs_x) + 'x'))
            elif counter == 0 and y == 1 and sign == '+':
                result.append(str(str(abs_x) + 'x'))
            elif y == 0 and sign == '-' and abs_x != 1 and counter == 0:
                result.append((f'-{abs_x}'))
            elif y == 0 and sign == '+' and abs_x != 1 and counter == 0:
                result.append((f'{abs_x}'))
            # first term - all other exponents
            elif counter == 0 and sign == '-' and abs_x == 1 and y > 1:
                result.append((f'-x^{y}'))
            elif counter == 0 and sign == '+' and abs_x == 1 and y > 1:
                result.append((f'x^{y}'))
            elif counter == 0 and sign == '-' and y > 1:
                result.append((f'-{abs_x}x^{y}'))
            elif counter == 0 and sign == '+' and y > 1:
                result.append((f'{abs_x}x^{y}'))

            # greater than first term - this is very poorly structured

            # following terms - zero exponent case - x is also zero!
            elif x == 0 and y == 0 and sign == '-':
                result.append(str('+ 0'))
            elif x == 0 and y == 0 and sign == '+':
                result.append(str('+ 0'))
            # following terms - zero exponent case - x is also zero!
            elif x == 0 and y == 0 and sign == '-' and abs_x != 1:
                result.append((f'- {abs_x}'))
            elif x == 0 and y == 0 and sign == '+' and abs_x != 1:
                result.append((f'+ {abs_x}'))

            # following terms - zero exponent case - x is not zero
            elif y == 0 and sign == '-':
                result.append(str('- ' + str(abs_x)))
            elif y == 0 and sign == '+':
                result.append(str('+ ' + str(abs_x)))

            # following terms - linear exponent case
            elif sign == '-' and y == 1 and abs_x == 1:
                result.append((f'- x'))
            elif sign == '+' and y == 1 and abs_x == 1:
                result.append((f'+ x'))
            # following terms - linear exponent case
            elif sign == '-' and y == 1:
                result.append((f'- {abs_x}x'))
            elif sign == '+' and y == 1:
                result.append((f'+ {abs_x}x'))
            # following terms - coef = 1 and exponent > 1 case
            elif sign == '-' and y > 1 and abs_x == 1:
                result.append((f'- x^{y}'))
            elif sign == '+' and y > 1 and abs_x == 1:
                result.append((f'+ x^{y}'))

            # following terms - exponent/coef greater than one case
            elif sign == '-':
                result.append((f'- {abs_x}x^{y}'))
            elif sign == '+':
                result.append((f'+ {abs_x}x^{y}'))

            # not sure if this gets reached?
            else:
                result.append((f'+ {abs_x}x^{y}'))
            counter += 1
        result = ' '.join(result)
        return result

############################################################
# Section 7: Python Packages
############################################################


def sort_array(list_of_matrices):
    result = []
    for x in list_of_matrices:
        if isinstance(x, np.ndarray):
            result.extend(np.ravel(x).astype(int))
        else:
            raise TypeError(
                f'Expected array. Input type was {type(x).__name__}')
    arr = np.array(result, dtype=int)
    return np.sort(arr)[::-1]


def POS_tag(sentence):
    sentence = sentence.lower()
    tokens = word_tokenize(sentence)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if
                       token not in stop_words and
                       token not in string.punctuation]

    tagged = nltk.pos_tag(filtered_tokens)
    return tagged

############################################################
# Section 8: Feedback
############################################################


# Just an approximation is fine.
feedback_question_1 = """
Approximately eight to ten hours. I wrote
all the code myself and
tried to learn as much as possible.
"""

feedback_question_2 = """
The first question that proved difficult was
transposing the matrix.
Conceptually, I understand it's simply
turning rows into columns,
but coding that proved to be difficult.
The other question that gave me trouble was
translating the polynomial back into english.
There were a ton of edge cases that gave me problems.
It was difficult to find all of them.
 After some time I managed to find them all.
"""

feedback_question_3 = """
This was a useful assignment. It was
very helpful to refresh myself with some of
the tools needed to parse data in python.
If I could modify it, I would advocate
for one simple question in each section.
Allow us to gain momentum, then give us
questions of increasing difficulty.
"""
