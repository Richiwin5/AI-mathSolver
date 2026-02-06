import random
import json
import pickle
import numpy as np
import nltk

from nltk.stem import WordNetLemmatizer
from keras.models import load_model
from reasoning_math_solver import solve_math
from math_extract import extract_math

lemmatizer = WordNetLemmatizer()

intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

model = load_model("chatbot_richiwin.h5")

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w: 
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    list_of_intents = intents_json['intents']
    
    # Check if any intent was predicted
    if len(intents_list) == 0:
        return "Sorry, I didn't understand that. Can you try rephrasing?"

    tag = intents_list[0]['intent']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])  # Make sure 'responses' exists in your JSON
            break
        
    return result


print("Great! Richiwin is running 🧠")

while True:
    user_input = input("You: ")

    # 1️⃣ Try math first
    math_expr = extract_math(user_input)

    if math_expr:
        answer = solve_math(math_expr)
        print("Richiwin:", answer)

    # 2️⃣ Otherwise use chatbot intents
    else:
        ints = predict_class(user_input)
        if ints:
            res = get_response(ints, intents)
        else:
            res = "I didn't understand that."
        print("Richiwin:", res)

# print("Great! Bot is Running")



# while True: 
#     message = input("")
#     ints = predict_class(message)
#     res = get_response(ints, intents)
#     print(res)


# import random
# import json 
# import pickle 
# import numpy as np
# import nltk 

# from nltk.stem import WordNetLemmatizer
# from keras.models import load_model

# lemmatizer = WordNetLemmatizer()

# intents = json.loads(open('intents.json').read())
# words = pickle.load(open('words.pkl', 'rb'))
# classes = pickle.load(open('classes.pkl', 'rb'))

# model = load_model("chatbot_richiwin.h5")

# def clean_up_sentence(sentence):
#     sentence_words = nlkt.word_tokenize(sentence)
#     sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
#     return sentence_words

# def bag_of_words(sentence):
#     sentence_words = clean_up_sentence(sentence)
#     bag = [0]*len(words)
#     for w in sentence_words:
#         for i, word in enumerate(words):
#             if word ==w: 
#                 bag[i]=1
#         return np.array(bag)   
    
# def predict_class(sentence):
#     bow = bag_of_words(sentence)
#     res = model.predict(np.array([bow]))[0]
    
#     ERROR_THRESHOLD = 0.25
#     results = [[i,r] for i, r in enumerate(res) if r>ERROR_THRESHOLD]
#     results.sort(key = lambda x:x[1], reverse = True)
#     return_list = []
#     for r in results:
#         return_list.append({'intent':classes[r[0]], 'probability':str(r[1])})
#     return return_list

# def get_response(intents_list, intents_json):
#     list_of_intents = intents_json['intents']
#     tag = intents_list[0]['intent']
#     for i in list_of_intents:
#         if i['tag']==tag:
#             result = random.choice(i[])
#             break 
        
#     return result 
# print("Great! Bot is Running")

# while True: 
#     message = input("")
#     ints = predict_class(message)
#     res = get_response(ints, intents)
#     print(res)
    
             
                