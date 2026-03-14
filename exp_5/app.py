from flask import Flask, render_template, request
import mysql.connector
import google.generativeai as genai
import os
from dotenv import load_dotenv
import json

load_dotenv()

app = Flask(__name__)


genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-flash-latest")


db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Insane10!",
    database="studentdb"
)

cursor = db.cursor()

def generate_sql_with_llm(user_query):
    prompt = f"""
You are a database assistant.

Convert the following natural language query into a MySQL SQL query.

Rules:
- Table name: students
- Columns: id, name, age, department, marks
- Only return valid SQL.
- Do not explain anything.
- Output in JSON format like:
{{ "sql": "SELECT * FROM students;" }}

User Query: {user_query}
"""

    response = model.generate_content(prompt)

    try:
        text = response.text.strip()
        json_data = json.loads(text)
        return json_data["sql"]
    except:
        return None


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    sql_query = None

    if request.method == "POST":
        user_query = request.form["query"]

        sql_query = generate_sql_with_llm(user_query)

        if sql_query:
            try:
                cursor.execute(sql_query)
                result = cursor.fetchall()
            except Exception as e:
                result = f"SQL Error: {e}"
        else:
            result = "Failed to generate SQL from LLM."

    return render_template("index.html", result=result, sql_query=sql_query)


if __name__ == "__main__":
    app.run(debug=True)