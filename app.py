from flask import Flask, render_template, request
import os

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/interpreter", methods=('GET', 'POST'))
def interpreter():
    text_output = ""
    is_processing = False
    
    if request.method == "POST":
        if 'video' in request.files:
            # In a real app, you would process the video here
            # For demo purposes, we're just returning a sample result
            text_output = "Hello! This is sample sign language interpretation text."
            is_processing = True
    
    return render_template("interpreter.html", 
                          text_output=text_output,
                          is_processing=is_processing)
