import random
import os
from flask import Flask, request, jsonify, Response, render_template
from transformers import pipeline
from transformers import AutoTokenizer, AutoModel,AutoModelForSequenceClassification
import numpy as np

app = Flask(__name__)

tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")
model = AutoModelForSequenceClassification.from_pretrained("facebook/bart-large-mnli")
zsc = pipeline(task='zero-shot-classification', tokenizer=tokenizer, model=model)


@app.route("/")
def hello():
    return "Helloooooooo playa, ya did it!!!!!!!!!"

@app.route("/pos_neg",methods=["POST"])
def get_pos_neg_score():
    data = request.json
    query = data["query"]
    #print(str(query))
    classes = ["positive","neutral","negative"]
    results = zsc(sequences=str(query), candidate_labels=classes, multi_class=False)
    print(results)
    n= np.argmax(results["scores"])
    sent={}
    for i in range(len(results["scores"])):
        sent[results["labels"][i]] = results["scores"][i]
    return jsonify(sent)
    
    
@app.route("/tonality",methods=["POST"])
def get_tonality():
    data = request.json
    query = data["query"]
    print(str(query))
    classes = ["curious","interested","irrelevant","uninterested","indifferent"]
    results = zsc(sequences=str(query), candidate_labels=classes, multi_class=False)
    n= np.argmax(results["scores"])
    sent={}
    for i in range(len(results["scores"])):
        sent[results["labels"][i]]= results["scores"][i]
    return jsonify(sent)
    
    
if __name__ == "__main__":
    app.run()
