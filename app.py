from datetime import datetime
import pandas as pd
import numpy as np
import joblib
import smtplib
import random
from flask import Flask, render_template, request, redirect, url_for, session, flash,jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
app.secret_key = 'supersecretkey'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///hackathon.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# User Model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)
    otp = db.Column(db.String(6), nullable=True)

# Problem Statements Model
class ProblemStatement(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    domain = db.Column(db.String(100), nullable=False)
    title = db.Column(db.String(255), nullable=False)
    description = db.Column(db.Text, nullable=False)
    github_link = db.Column(db.String(255))
    youtube_link = db.Column(db.String(255))
    resource_link = db.Column(db.String(255))


class Rating(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    solution_id = db.Column(db.Integer, db.ForeignKey('solution.id'), nullable=False)
    rating = db.Column(db.Integer, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    user = db.relationship('User', backref='ratings')
    solution = db.relationship('Solution', backref='ratings')


class Solution(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    problem_id = db.Column(db.Integer, db.ForeignKey('problem_statement.id'), nullable=False)
    comment = db.Column(db.Text, nullable=False)

    # Establish relationship to get user details
    user = db.relationship('User', backref='solutions')


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Send OTP via Email
def send_otp(email, otp):
    sender_email = "daminmain@gmail.com"
    sender_password = "kpqtxqskedcykwjz"
    subject = "Your OTP Code"
    body = f"Your OTP is: {otp}"
    
    message = f"Subject: {subject}\n\n{body}"
    with smtplib.SMTP('smtp.gmail.com', 587) as server:
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, email, message)


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    # Fetch user data from the database
    user = User.query.get(current_user.id)
    
    if request.method == 'POST':
        new_username = request.form['username']
        new_email = request.form['email']
        new_password = request.form['password']
        
        # Update the user data in the database
        if new_username:
            user.username = new_username
        if new_email:
            user.email = new_email
        
        db.session.commit()
        flash("Profile updated successfully!", "success")
        return redirect(url_for('profile'))

    return render_template('profile.html', user=user)


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        otp = str(random.randint(100000, 999999))
        
        user = User(username=username, email=email, password=password, otp=otp)
        db.session.add(user)
        db.session.commit()
        send_otp(email, otp)
        flash("OTP sent to your email!", "info")
        return redirect(url_for('verify_otp', email=email))
    return render_template('register.html')

@app.route('/verify_otp', methods=['GET', 'POST'])
def verify_otp():
    email = request.args.get('email')
    if request.method == 'POST':
        entered_otp = request.form['otp']
        user = User.query.filter_by(email=email, otp=entered_otp).first()
        if user:
            user.otp = None
            db.session.commit()
            flash("Registration successful!", "success")
            return redirect(url_for('login'))
        else:
            flash("Invalid OTP. Try again.", "danger")
    return render_template('verify_otp.html', email=email)
@property
def average_rating(self):
        """Calculate and return the average rating."""
        if self.total_ratings == 0:
            return 0  # Default if no ratings yet
        return self.rating_sum / self.total_ratings
    

@app.route('/rate-solution', methods=['POST'])
@login_required
def rate_solution():
    data = request.get_json()
    solution_id = data.get('solution_id')
    rating = data.get('rating')

    solution = Solution.query.get(solution_id)
    if not solution:
        return jsonify({'error': 'Solution not found'}), 404

    # Update rating sum and count
    solution.rating_sum += rating
    solution.total_ratings += 1
    db.session.commit()

    return jsonify({'message': 'Rating submitted!', 'average_rating': solution.average_rating})



@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email, password=password).first()
        if user:
            login_user(user)
            return redirect(url_for('dashboard'))
        else:
            flash("Invalid credentials!", "danger")
    return render_template('login.html')

@app.route('/dashboard')
@login_required
def dashboard():
    domains = [d[0] for d in db.session.query(ProblemStatement.domain).distinct().all()]
    print("Debug: Domains fetched:", domains)  # Debugging output
    return render_template('dashboard.html', domains=domains)

from urllib.parse import unquote_plus

@app.route('/domain/<path:domain>')
@login_required
def view_statements(domain):
    domain = unquote_plus(domain)  # Ensure spaces are properly decoded
    problems = ProblemStatement.query.filter_by(domain=domain).all()
    print(f"Debug: Received domain - {domain}")  
    print(f"Debug: Found {len(problems)} problems for domain")  
    return render_template('statements.html', problems=problems, domain=domain)

import matplotlib.pyplot as plt
import io
from flask import Response
@app.route('/graph_analysis')
@login_required
def graph_analysis():
    # Graph 1: Number of solutions posted by each user
    user_solution_counts = db.session.query(Solution.user_id, db.func.count(Solution.id)).group_by(Solution.user_id).all()
    user_names = [User.query.get(user_id).username for user_id, _ in user_solution_counts]
    solution_counts = [count for _, count in user_solution_counts]
    
    plt.figure(figsize=(10,6))
    plt.bar(user_names, solution_counts, color='skyblue')
    plt.xlabel('Users')
    plt.ylabel('Number of Solutions')
    plt.title('Number of Solutions Posted by Each User')
    
    # Save the plot to a BytesIO buffer
    img_io = io.BytesIO()
    plt.savefig(img_io, format='png')
    img_io.seek(0)
    plt.close()  # Close the plot to free up memory
    
    # Return the graph image as a response
    return Response(img_io, mimetype='image/png')

@app.route('/post_solution/<int:problem_id>', methods=['POST'])
@login_required
def post_solution(problem_id):
    problem = ProblemStatement.query.get(problem_id)  # Ensure problem exists
    if not problem:
        flash("Problem statement not found!", "danger")
        return redirect(url_for('dashboard'))

    comment = request.form['comment']
    solution = Solution(user_id=current_user.id, problem_id=problem_id, comment=comment)
    db.session.add(solution)
    db.session.commit()

    flash("Solution posted successfully!", "success")
    return redirect(url_for('view_statements', domain=problem.domain))


@app.route('/view_solutions/<int:problem_id>')
@login_required
def view_solutions(problem_id):
    problem = ProblemStatement.query.get(problem_id)
    if not problem:
        flash("Problem statement not found!", "danger")
        return redirect(url_for('dashboard'))

    solutions = Solution.query.filter_by(problem_id=problem_id).all()
    return render_template('solutions.html', problem=problem, solutions=solutions)

from markupsafe import Markup

@app.template_filter('urlize')
def urlize_filter(text):
    import re
    url_pattern = re.compile(r"(https?://\S+)")
    return Markup(re.sub(url_pattern, r'<a href="\1" target="_blank">\1</a>', text))

responses = {
    "hi": ["Hello! How can I assist you today?", "Hi there! What do you need help with?"],
    "hello": ["Hey! How can I help?", "Hello! What would you like to know?"],
    "how to register": "To register, click on 'Register', fill in your details (username, email, password), and submit. You will receive an OTP via email. Enter the OTP to verify your account and complete the registration process.",
    "how to log in": "Go to the login page, enter your registered email and password, then click 'Login'. If you forgot your password, use the 'Forgot Password' option to reset it.",
    "how to post a solution": "After logging in, navigate to the problem statement page, enter your solution in the text box, and click 'Post Solution'.",
    "how to view solutions": "Visit the problem statements section, select a problem, and click 'View Solutions' to see responses from other users.",
    "how to update my profile": "Go to your profile page and click 'Edit Profile'. You can update your name, email, and password. Save changes to update your details.",
    "how to manage my account settings": "Navigate to 'Account Settings' to modify your password, notification preferences, and other account details.",
    "what are problem domains": "Problem domains categorize different challenges into areas like AI, Web Development, and Data Science to help users find relevant problems.",
    "how to filter problems by domain": "Use the 'Filter by Domain' option on the dashboard to view problems related to a specific field of interest.",
    "how to track my solutions": "Go to your profile, where you can see a list of problems you've contributed to and track their status.",
    "how to contact support": "Visit the 'Contact Us' page, submit a support request, and our team will assist you as soon as possible.",
    "thank you": ["You're welcome! 😊", "Happy to help!", "Anytime! Let me know if you need more assistance."]
}

@app.route('/chatbot', methods=['GET', 'POST'])
def chatbot():
    if request.method == 'POST':
        user_message = request.form['message'].lower().strip()
        
        # Check if the exact user input exists in responses
        if user_message in responses:
            response = random.choice(responses[user_message]) if isinstance(responses[user_message], list) else responses[user_message]
        else:
            # Default fallback response for unknown questions
            response = "I'm not sure about that. Can you rephrase or ask something else?"
        
        return jsonify({'response': response})
    
    return render_template('chatbot.html')  # Load chatbot UI


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

if __name__ == '__main__':
    with app.app_context():
        db.create_all()

        # Load dataset from CSV
        try:
            df = pd.read_csv("hackathon_dataset.csv")

            for _, row in df.iterrows():
                # Check if record already exists
                existing_problem = ProblemStatement.query.filter_by(title=row["Problem Statement"]).first()
                
                if not existing_problem:
                    new_problem = ProblemStatement(
                        domain=row["Domain"],
                        title=row["Problem Statement"],
                        description=row["Description"],
                        github_link=row.get("GitHub Link", None),  # Handle missing data
                        youtube_link=row.get("YouTube Link", None),
                        resource_link=row.get("Other Resource Link", None)
                    )
                    db.session.add(new_problem)

            db.session.commit()
            print("Dataset imported successfully!")

        except Exception as e:
            print(f"Error importing dataset: {e}")

    app.run(debug=True)
