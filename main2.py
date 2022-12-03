print("--"*33)
from nltk.stem.lancaster import LancasterStemmer
import tflearn
import tensorflow
import numpy as np
import mytokenizer
import random as rd
import json
import pickle
print("--"*33)

"""
prerequisites = [
    "pickle <inbuilt>",
    "json <inbuilt>",
    "random <inbuilt>",
    "numpy",
    "tensorflow"
    "tflearn"
    "nltk"
]
"""

stemmer = LancasterStemmer()

with open("intents.json") as file:
    data = json.load(file)

words = []
labels = []
docs_x = []
docs_y = []

for intent in data["intents"]:
    for pattern in intent["tag"]:
        wrds = mytokenizer.tokenizerx(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])

    if intent["tag"] not in labels:
        labels.append(intent["tag"])

words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))

labels = sorted(labels)

training = []
output = []

output_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = []

    wrds = [stemmer.stem(w) for w in doc]

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)
    output_row = output_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)

# deeplearnin sector
training = np.array(training)
output = np.array(output)
tensorflow.compat.v1.reset_default_graph()
# trainign the neural network
net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)
model = tflearn.DNN(net)

model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
model.save("model2.tflearn")

# deeplearnin sector endblock

def bag_of_words(s, words):
    bag = [
        0 for _ in range(len(words))
    ]
    s_words = mytokenizer.tokenizerx(s)
    s_words = [
        stemmer.stem(word.lower()) for word in s_words
    ]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    return np.array(bag)


def chat():
    endtask = ("quit", "q", "end")
    print("welcome to a something cookie shop")
    print(f"start talking to the helpline bot🤖 type {endtask}")
    while True:
        inp = input("you: ")
        if inp.lower() in endtask:
            break

        results = model.predict([bag_of_words(inp, words)])
        print(results)
        results_index = np.argmax(results) # gives out the max result
        tag = labels[results_index]
        for tg in data["intents"]:
            if (tg['tag'] == tag):
                responses = tg['responses']
                to_user = rd.choice(responses)
                print(to_user)

chat()