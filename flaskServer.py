from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, send
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user  # Added current_user
import boto3
from flask_bcrypt import Bcrypt
from uuid import uuid4
from datetime import datetime
import logging
import sys
from flask import request, redirect, url_for, flash
import eventlet

app = Flask(__name__)
app.config['SECRET_KEY'] = 'SUPER_SECRET_KEY4321'
socketio = SocketIO(app, cors_allowed_origins="*")

# Create a DynamoDB client
dynamodb = boto3.resource('dynamodb', region_name='eu-north-1')

# Reference to the Users and ChatHistory tables
users_table = dynamodb.Table('Users')
chat_table = dynamodb.Table('ChatHistory')

# Encryption
bcrypt = Bcrypt(app)

# User login management
login_manager = LoginManager()
login_manager.init_app(app)

# Logging configuration
logging.basicConfig(level=logging.INFO, stream=sys.stdout, 
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

class User(UserMixin):
    def __init__(self, username, email, password):
        self.username = username
        self.email = email
        self.password = password

    @staticmethod
    def get_user_by_id(user_id):
        # Retrieve user from DynamoDB
        response = users_table.get_item(Key={'username': user_id})
        if 'Item' in response:
            data = response['Item']
            return User(data['username'], data['email'], data['password'])
        return None

    def get_id(self):
        return self.username

    @staticmethod
    def create_user(username, email, password):
        users_table.put_item(
            Item={
                'username': username,
                'email': email,
                'password': password
            }
        )

@login_manager.user_loader
def load_user(user_id):
    return User.get_user_by_id(user_id)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Fetch the user from DynamoDB
        user_data = User.get_user_by_id(username)

        if user_data and bcrypt.check_password_hash(user_data.password, password):
            user = User(username=user_data.username, email=user_data.email, password=user_data.password)
            login_user(user)
            flash('Login successful!', 'success')
            return redirect(url_for('chat'))
        else:
            flash('Login failed. Please check your credentials.', 'danger')
            return redirect(url_for('login'))

    return render_template('login.html')

@app.route('/register', methods=['GET','POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')

        if User.get_user_by_id(username):
            flash('Username already exists!', 'danger')
            return redirect(url_for('register'))

        User.create_user(username, email, hashed_password)
        flash('Registration successful!', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/chat')
def chat():
    return render_template('chat.html')

@app.route('/log_chat', methods=['POST'])
def log_chat():
    username = 'Guest'
    if current_user.is_authenticated:
        username = current_user.username

    data = request.get_json()
    message = data.get('message')

    timestamp = datetime.now().isoformat()

    chat_table.put_item(
        Item={
            'username': username,
            'timestamp': timestamp,
            'message': message
        }
    )

    return jsonify({'status': 'success', 'message': 'Message logged'}), 200

@socketio.on('message')
def handle_message(message):
    logger.info(f"Message received: {message}")
    response = "Ok, I'll work on it buddy !"

    message_id = str(uuid4())
    timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')

    username = 'Guest'
    if current_user.is_authenticated:
        username = current_user.username

    try:
        chat_table.put_item(
            Item={
                'username': username,
                'message_id': message_id,
                'timestamp': timestamp,
                'user_message': message,
                'server_response': response,
            }
        )
        logger.info(f"Chat logged successfully: {message_id}")
    except Exception as e:
        logger.error(f"Error logging chat to DynamoDB: {e}")

    send(response, broadcast=True)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect('/login')

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
