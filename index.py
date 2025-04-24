from flask import Flask, render_template, request

app = Flask(__name__)

last_data = {
    "name": "John Doe",
    "age": 30
}

@app.route("/")
def hello_world():
    return render_template("index.html")

@app.route("/testform", methods=('GET', 'POST'))
def test_form():
    if request.method == "GET":
        return render_template("testform.jinja", name=last_data["name"], age=last_data["age"])
    if request.method == "POST":
        name = request.form.get("name")
        age = request.form.get("age")
        last_data["name"] = name
        last_data["age"] = age
        print(f"Name: {name}, Age: {age}")
        return render_template("testform.jinja", name=name, age=age)

#Run command
#python -m flask --app web_app/index run