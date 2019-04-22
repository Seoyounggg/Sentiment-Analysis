import re

with open('./Data/ratings_test.txt', 'r', encoding='utf-8') as input_1:
    a = input_1.readlines()
    a = a[1:]

print(a[0])

b = re.split('\t', a[1])

print(b)