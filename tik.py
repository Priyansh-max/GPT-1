import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F

enc = tiktoken.get_encoding('gpt2')
print(enc.n_vocab)

print(enc.encode("h i i t h e r e"))
print(enc.encode("hi"))
print(enc.encode("hii")) #here h --> 71 and ii --> 4178
print(enc.encode("hiiiiii")) #here again h--> 71 and 3 pairs of ii therefore 3 times 4178
print(enc.encode("hiiiii")) #here again h--> 71 and 1 pair of ii --> 4178 and iii --> 15479 it forms subword of max length 3 and encodes it
print(enc.encode("hiiiiiii"))

"""hiiiii --> hence final encodes h , ii , iii hence encode array is of length 3
"""
context = torch.zeros((4, 4)[1], dtype=torch.long)
print(context)        

h = input("hii : ")
print(h)     
