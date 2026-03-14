from flask import Flask, render_template, request, redirect, session
from werkzeug.security import generate_password_hash, check_password_hash
from db_config import mysql

app = Flask(__name__)

# ---------------- SECRET KEY ----------------
app.secret_key = 'secret123'


# ---------------- MYSQL CONFIG ----------------
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'

# 🔴 PUT YOUR MYSQL PASSWORD HERE 🔴
app.config['MYSQL_PASSWORD'] = 'Insane10!'

app.config['MYSQL_DB'] = 'flask_auth_db'

mysql.init_app(app)


# ---------------- HOME ----------------
@app.route('/')
def home():
    return "Database Connected Successfully"


# ---------------- SIGNUP ----------------
@app.route('/signup', methods=['GET', 'POST'])
def signup():

    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        hashed_password = generate_password_hash(password)

        cursor = mysql.connection.cursor()

        cursor.execute(
            "INSERT INTO users(username, email, password) VALUES(%s,%s,%s)",
            (username, email, hashed_password)
        )

        mysql.connection.commit()
        cursor.close()

        return "User Registered Successfully"

    return render_template('signup.html')


# ---------------- LOGIN ----------------
@app.route('/login', methods=['GET', 'POST'])
def login():

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        cursor = mysql.connection.cursor()
        cursor.execute(
            "SELECT * FROM users WHERE username=%s",
            (username,)
        )

        user = cursor.fetchone()
        cursor.close()

        if user:
            stored_password = user[3]

            if check_password_hash(stored_password, password):
                session['username'] = username
                return redirect('/dashboard')
            else:
                return "Wrong Password"
        else:
            return "User not found"

    return render_template('login.html')


# ---------------- DASHBOARD ----------------
@app.route('/dashboard')
def dashboard():

    if 'username' in session:
        return render_template(
            'dashboard.html',
            username=session['username']
        )

    return redirect('/login')


# ---------------- PROFILE UPDATE ----------------
@app.route('/profile', methods=['GET', 'POST'])
def profile():

    if 'username' not in session:
        return redirect('/login')

    if request.method == 'POST':
        new_username = request.form['username']
        new_email = request.form['email']

        cursor = mysql.connection.cursor()

        cursor.execute(
            "UPDATE users SET username=%s, email=%s WHERE username=%s",
            (new_username, new_email, session['username'])
        )

        mysql.connection.commit()
        cursor.close()

        session['username'] = new_username

        return "Profile Updated Successfully"

    return render_template('profile.html')


# ---------------- VIEW GRADES ----------------
@app.route('/grades')
def grades():

    if 'username' not in session:
        return redirect('/login')

    cursor = mysql.connection.cursor()

    # Get user id
    cursor.execute(
        "SELECT id FROM users WHERE username=%s",
        (session['username'],)
    )
    user = cursor.fetchone()
    user_id = user[0]

    # Get grades
    cursor.execute(
        "SELECT subject, marks, grade FROM grades WHERE user_id=%s",
        (user_id,)
    )

    grades_data = cursor.fetchall()
    cursor.close()

    return render_template(
        'grades.html',
        grades=grades_data
    )


# ---------------- LOGOUT ----------------
@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect('/login')


# ---------------- RUN SERVER ----------------
if __name__ == "__main__":
    app.run(debug=True)
