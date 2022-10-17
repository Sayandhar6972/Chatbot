
from pydoc import doc
import random
import json
import pickle
import numpy as np


import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

lemmatizer = WordNetLemmatizer()

#nltk.download('wordnet')
#nltk.download('omw-1.4')
intents = json.loads(open('intents.json').read()) #json read 

words = []
classes = []
documents = []
ignore_letters = ['?','.','!',',']  

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)    
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
#print(documents)
#print(classes)
#print(words)

words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))

#print(words)

# <-------------- Training Model ----------->

training = []
output_empty =[0] * len(classes)
#print(output_empty)

for document in documents:                  #documents-----> XY_Train Dataset
    bag=[]
    word_patterns=document[0]
    word_patterns=[lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    #print(words)
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
    
   # print(document[1])
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    #print(output_row)
    training.append([bag,output_row])
    #print(training)



random.shuffle(training)
training = np.array(training)

#print(training)

train_x = list(training[:, 0])
train_y = list(training[:, 1])

#print("Train_X : ",train_x,end="\n\n")
#print("Train_Y: ",train_y,end="\n\n")



