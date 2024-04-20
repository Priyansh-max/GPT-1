import torch
import torch.nn as nn
from torch.nn import functional as F
import pandas as pd

#imported data

df = pd.read_csv("text.csv")
text = df["text"].str.cat(sep="\n")
# print("Length of text : ",len(text))


#already existed char level encoder with vocab size of 256

# tokens1 = text.encode("utf-8") #it encoded char to integer from 0 to 255 with vocab size of 256
# tokens1 = list(map(int , tokens1)) 
# #utf encoding is just another character level encoding with a vocab size of 256 .......i.e
# #it contains 255 different characters in its vocab
# print(len(tokens1), "for utf-8")
# #a start from 97
#z is at 122
# print(chr(122))

# print(chr(0) , chr(1) , chr(2) , chr(3) , chr(4) ," ..............." , chr(256))


#our created encoder with vocab size of 28
chars = sorted(list(set(text)))
vocab_size = len(chars)
# print("Size of vocab : ",vocab_size)

stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l]) 

tokens = list(encode(text)) #it encoded char to integer from 0 to 27 with vocab size of 28
print(len(tokens) , "our version")
# print(encode("e"))
# print(chars)
# print(decode([27]))

print("inital")
print(stoi)
print(itos)


#function to find the frequency of frequent consecutive pairs 
def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]): # Pythonic way to iterate consecutive elements
        counts[pair] = counts.get(pair, 0) + 1
    return counts

#utf-8
# stats1 = get_stats(tokens1)

# #the one we made
# stats = get_stats(tokens)

# print("utf-8")
# print(sorted(((v,k) for k,v in stats1.items()), reverse=True))

# print("our version")
# print(sorted(((v,k) for k,v in stats.items()), reverse=True))


# print("utf-8")

# top_pair1 = max(stats1, key=stats1.get)
# print(top_pair1)

# print("our version")

# top_pair = max(stats, key=stats.get)
# print(top_pair)

def merge(ids, pair, idx):
  # in the list of ints (ids), replace all consecutive occurences of pair with the new token idx
  newids = []
  i = 0
  while i < len(ids):
    # if we are not at the very last position AND the pair matches, replace it
    if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
      newids.append(idx)
      i += 2
    else:
      newids.append(ids[i])
      i += 1
  return newids

#utf-8
# max_vocab1 = 280 
# num_merges1 = max_vocab1 - 256
# new_list1 = list(tokens1)

# merges1 = {}
# for i in range(num_merges1):
#     new_count = get_stats(new_list1)
#     pair = max(new_count , key=new_count.get)
#     idx = 256 + i
#     print(f"merging {pair} into a new token {idx}")
#     new_list1 = merge(new_list1 , pair , idx)
#     merges1[pair] = idx

    
#our version
  
max_vocab = 31 
num_merges = max_vocab - 28
new_list = list(tokens)

merges = {}
for i in range(num_merges):
    new_count = get_stats(new_list)
    pair = max(new_count , key=new_count.get)
    idx = 28 + i
    print(f"merging {pair} into a new token {idx}")
    new_list = merge(new_list , pair , idx)
    merges[pair] = idx
    
print(merges)

for key, value in merges.items():
    var1 , var2 = key
    new_var1 = decode([var1])
    new_var2 = decode([var2])
    new_idx = new_var1+new_var2
    itos[value] = new_idx

print(itos)

print(decode([30]))
    
    
    






