print("--"*33)
import pickle
import json
import random as rd
import mytokenizer
import numpy as np
import tensorflow
import tflearn
from nltk.stem.lancaster import LancasterStemmer
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
try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
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

    training = np.array(training)
    output = np.array(output)
    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

# resetting raph
tensorflow.compat.v1.reset_default_graph()
# trainign the neural network
net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 16)
net = tflearn.fully_connected(net, 16)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)
model = tflearn.DNN(net)

# if already trained
try:
    model.load("model.tflearn")
except:
    # if not trained
    model.fit(training, output, n_epoch=5000, batch_size=8, show_metric=True)
    model.save("model.tflearn")


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
        results_index = np.argmax(results)
        tag = labels[results_index]

        some_humor = ["what's up", "what up", "wat up"]
        if ("sell" in inp or inp == "sell" or "much" in inp or "cookie" in inp or "cookies" in inp or "menu" in inp):
            to_user = "We sell chocolate chip cookies for $2"
            print(to_user)

        elif (inp == "hello" or "hey" in inp):
            responses_to_hello = [
                "Hello!",
                "Good to see you again!",
                "Hi there, how can I help?"
            ]
            to_user = rd.choice(responses_to_hello)
            print(to_user)
        elif "age" in inp or "old" in inp:
            to_user = "I'm 16 years old"
            print(to_user)
        elif inp in some_humor:
            to_user = "creating a black hole to end all life"
            print(to_user)

        elif ("open" in inp or "time" in inp):
            to_user = "We are open 7am-4pm Monday-Friday!"
            print(to_user)

        elif (results[0][results_index] > 0.9):
            for tg in data["intents"]:
                if (tg['tag'] == tag):
                    responses = tg['responses']
            to_user = rd.choice(responses)
            print(to_user)

        else:
            to_user = "I didn't quite get that"
            print(to_user)


chat()
# made updates to mytokenizer, re train this model later