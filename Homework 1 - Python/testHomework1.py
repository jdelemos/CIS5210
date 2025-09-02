import numpy as np
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger_eng')

def concatenate(seqs):
    return [y for x in seqs for y in x]

x = concatenate(["abc", (0, [0])])

print(x)


def transpose(matrix):
    rows = len(matrix)
    columns = len(matrix[0])
    transposed_matrix = [[0 for _ in range(rows)] for _ in range(columns)]
    for i in range(rows): 
        for j in range(columns): 
            transposed_matrix[j][i] = matrix[i][j]
    return transposed_matrix
                

transpose([[1, 2, 3]])

def copy(seq):
    new_copy = list(seq)
    print(new_copy)
    seq[0] = 1
    print(seq)

    assert new_copy == seq

copy([1,2,3])


def all_but_last(seq):
    return seq[0:-1]

def every_other(seq):
    return (seq[::2])

every_other([0,1,2,3,4,5,6])

print('debug')

def prefixes(seq):
    for i in range(len(seq)+1):
        yield seq[:i]

def suffixes(seq):
    for i in range(len(seq)+1):
        yield seq[i:]
        
def slices(seq):
    for i in range(len(seq)):
        for j in range(len(seq)+1):
            yield seq[i:j]

def normalize(text):
    new_text = text.lower().strip().split()
    return ' '.join(new_text)
    


normalize('Hello     how are you')
def no_vowels(text):
   new_text = list(x for x in text if x not in ['A','a', 'E', 'e','I', 'i', 'O','o', 'U','u'])
   return ''.join(new_text)
  

no_vowels('aeiou')

#this is a very stupid solution. I'm sorry to the university
def digits_to_words(text):
    result = []
    new = list(text)
    print(new)
    for x in range(len(new)):
        print(new[x])
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
            #kind of just a dummy place holder
            y = 0 
    return ' '.join(result)

#digits_to_words('hello 1234')


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
    print(''.join(name))



to_mixed_case('____ahfja___jafj')

class Polynomial():
    # I don't think this is so trivial
    def __init__(self, polynomial): 
        self.polynomial = tuple(polynomial)
        
    def get_polynomial(self):
        return self.polynomial
    
    def __neg__(self): 
        new_polynomial = []
        for x,y in self.polynomial:
            new_polynomial.append((-x,y))
        return Polynomial(new_polynomial)
    
    def __add__(self, other):
        return Polynomial(self.polynomial+other.polynomial)
    
    def __sub__(self, other):
        new_poly = other.__neg__()
        return Polynomial(self.polynomial+ new_poly.polynomial)
    
    def __mul__(self, other):
        new_poly = []
        for x,y in self.polynomial:
            for f,g in other.polynomial:
                new_poly.append((x*f, y+g))
        return Polynomial(tuple(new_poly))
    
    def __call__(self, x):
        new_poly = []
        for f,g in self.polynomial:
            new_poly.append(f*(x**g))
        return sum(new_poly)
    
    #this took me 3 fucking hours
    def simplify(self):
        #declare an empty list
        result = []
        #defend against empty lists
        if len(self.polynomial) == 0:
            # learned today this means a tuple with one term
            self.polynomial = ((0,0),)
            return
        #meet the immutability requirements
        new_list = list(self.polynomial)
        #sort it descending per the requirements - this is key 
        new_list.sort(key = lambda sorting:sorting[1], reverse=True)
        #grab the comparison variables
        exp = new_list[0][1]
        cur = new_list[0][0]
        #loop through with a slice comparing the later variables with the comparison variables
        for x,y in new_list[1:]:
            if y == exp: 
                cur+=x
            #update comparison variables if they dont match
            else:
                result.append((cur,exp))
                cur = x
                exp = y  
        #append the final pair
        result.append((cur,exp))
        #defend against zero pairs
        final_list = []
        for x,y in result: 
            if x != 0:
                final_list.append((x,y)) 
        #if empty, add a pair of zeros              
        if len(final_list) == 0:
            self.polynomial = ((0,0),)
            return
        #turn it back into a tuple
        self.polynomial = tuple(final_list)
        


            

            
            

    #notes to remember, join only works with strings! this is totally fucking ridiculous. This was a literal nightmare
    def __str__(self):
        
        result = []
        counter = 0
        new_list = list(self.polynomial)
        for x,y in new_list:
            #determine positive or negative value 
            if x < 0:
                sign = '-'
            else: 
                sign = '+'
            if x == 0: 
                sign = '+'
            #after sign has been gathered, take the abs value, save for formatting
            abs_x = abs(x)
            #this was an absolute mess

            #first term - zero exponent
            if x == 0 and counter == 0 and y == 0 and sign == '-': 
                result.append('0')
            elif x == 0 and counter == 0 and y == 0 and sign == '+': 
                result.append('0')

            #first term - zero exponent
            elif counter == 0 and y == 0 and sign == '-' and abs_x == 1: 
                result.append(str(-1))
            elif counter == 0 and y == 0 and sign == '+' and abs_x == 1: 
                result.append(str(1))
            
            #first term - linear value exponent
            elif counter == 0 and y == 1 and sign == '-' and abs_x == 1: 
                result.append(str('-x'))
            elif counter == 0 and y == 1 and sign == '+' and abs_x == 1: 
                result.append(str('x'))
                

            #first term - linear value exponent
            elif counter == 0 and y == 1 and sign == '-': 
                result.append(str('-'+str(abs_x)+'x'))
            elif counter == 0 and y == 1 and sign == '+': 
                result.append(str(str(abs_x)+'x'))

            elif y == 0 and sign == '-' and abs_x != 1 and counter == 0:
                result.append((f'-{abs_x}'))
            elif y == 0 and sign == '+' and abs_x != 1 and counter == 0:
                result.append((f'{abs_x}'))
     

            #first term - all other exponents
            elif counter == 0 and sign == '-' and abs_x == 1 and y > 1: 
                result.append((f'-x^{y}'))
            elif counter == 0 and sign == '+' and abs_x == 1 and y > 1: 
                result.append((f'x^{y}'))
            
            elif counter == 0 and sign == '-' and y > 1: 
                result.append((f'-{abs_x}x^{y}'))
            elif counter == 0 and sign == '+' and y > 1: 
                result.append((f'{abs_x}x^{y}'))


            #greater than first term - this is very poorly structured

            #following terms - zero exponent case - x is also zero!
            elif x == 0 and y == 0 and sign == '-':
                result.append(str('+ 0'))
            elif x == 0 and y == 0 and sign == '+':
                result.append(str('+ 0'))
            
            #following terms - zero exponent case - x is also zero!
            elif x == 0 and y == 0 and sign == '-' and abs_x != 1:
                result.append((f'- {abs_x}'))
            elif x == 0 and y == 0 and sign == '+' and abs_x != 1:
                result.append((f'+ {abs_x}'))

            #following terms - zero exponent case - x is not zero 
            elif y == 0 and sign == '-':
                result.append(str('- ' + str(abs_x)))
            elif y == 0 and sign == '+':
                result.append(str('+ ' +str(abs_x)))

            #following terms - linear exponent case
            elif sign == '-' and y == 1 and abs_x == 1:
                result.append((f'- x'))
            elif sign == '+' and y == 1 and abs_x == 1:
                result.append((f'+ x'))
            
             #following terms - linear exponent case
            elif sign == '-' and y == 1:
                result.append((f'- {abs_x}x'))
            elif sign == '+' and y == 1:
                result.append((f'+ {abs_x}x'))
            
            #following terms - linear exponent case
            elif sign == '-' and y >1 and abs_x == 1:
                result.append((f'- x^{y}'))
            elif sign == '+' and y > 1 and abs_x == 1:
                result.append((f'+ x^{y}'))


            #following terms - exponent greater than one case
            elif sign == '-':
                result.append((f'- {abs_x}x^{y}'))
            elif sign == '+':
                result.append((f'+ {abs_x}x^{y}'))

            #not sure if this gets reached? 
            else: 
                result.append((f'+ {abs_x}x^{y}'))
            counter +=1
        result = ' '.join(result)
        return result

        
        
        

p = Polynomial([(2, 1), (1, 0)])
p.get_polynomial()
q = -p + (p * p)
#[p(x) for x in range(5)]
q.simplify()
q.get_polynomial(); print(str(q))
p = Polynomial([(0, 1), (2, 3)])
print(str(p)); print(str(p * p)); print(str(-p * p))

q = Polynomial([(0,1), (5,6)])
print(str(q)); print(str(q * q)); print(str(-q * q))

tests = [
    Polynomial([(0,0)]),
    Polynomial([(0,1), (0,2)]),
    Polynomial([(0,1), (2,3)]),
    Polynomial([(4,2), (0,1), (0,4)]),
    Polynomial([(16,0), (2,1)]),
    Polynomial([(-9,0), (1,2)]),
    Polynomial([(1,0), (1,1)]),
    Polynomial([(-1,0), (1,1)]),
    Polynomial([(1,1)]),
    Polynomial([(-1,1)]),
    Polynomial([(5,1)]),
    Polynomial([(-7,1)]),
    Polynomial([(1,5)]),
    Polynomial([(-1,4)]),
    Polynomial([(3,2), (5,0)]),
    Polynomial([(3,2), (-5,0)]),
    Polynomial([(0,0), (0,1), (4,2)]),
    Polynomial([(2,3), (0,2), (-1,1)]),
    Polynomial([(2,3), (2,3)]),
    Polynomial([(1,9), (-1,8), (2,1), (-2,0)]),
    Polynomial([(2,4), (1,3), (-1,2), (1,1), (-1,1)]),
    Polynomial([(0,3), (-2,2), (0,1), (5,0)]),
    Polynomial([(-2,5), (1,2), (1,1)]),
    Polynomial([(1,6), (-3,2), (4,0)]),
    Polynomial([(7,0)]),
    Polynomial([(-7,0)]),
    Polynomial([(1,0)]),
    Polynomial([(-1,0)])
]

for i, q in enumerate(tests, start=1):
    print(f"Test {i}")
    print(str(q))
    print(str(q * q))
    print(str(-q * q))
    print()


def sort_array(list_of_matrices):
    result = []
    for x in list_of_matrices:
        if isinstance(x, np.ndarray): 
            result.extend(np.ravel(x).astype(int))
        else: 
            raise TypeError(f'Expected array. Input type was {type(x).__name__}')   
    arr = np.array(result, dtype = int)
    return np.sort(arr)[::-1]

matrix1 = np.array([[1, 2], [3, 4]])
matrix2 = np.array([[5, 6, 7], [7, 8, 9], [0, -1, -2]])
sort_array([matrix1, matrix2])


def POS_tag(sentence):
    sentence = sentence.lower()
    tokens = word_tokenize(sentence)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token not in stop_words and token not in string.punctuation]

    tagged = nltk.pos_tag(filtered_tokens)
    return tagged


sentence = 'The force will be with you. Always.'
POS_tag(sentence)