from flask import Flask, render_template, request
from flask_socketio import SocketIO, send
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user
#import your_robot_script  # This is where your robot's Python logic is handled
import boto3
# libraries to manage credentials
from flask import request, redirect, url_for, flash
from flask_bcrypt import Bcrypt
from uuid import uuid4
import eventlet
from datetime import datetime
import logging
import sys


  
app = Flask(__name__)
app.config['SECRET_KEY'] = 'SUPER_SECRET_KEY4321'
socketio = SocketIO(app, cors_allowed_origins="*")

# Create a DynamoDB client
dynamodb = boto3.resource('dynamodb', region_name='eu-north-1')

# Reference to the Users and ChatHistory tables
users_table = dynamodb.Table('Users')
chat_table = dynamodb.Table('ChatHistory')

# management of the encryption of the app
bcrypt = Bcrypt(app)

# User login management
login_manager = LoginManager()
login_manager.init_app(app)

# Manage print into log files
logging.basicConfig(level=logging.INFO, stream=sys.stdout, 
                    format='%(asctime)s %(levelname)s %(message)s')
### DEBUG TRACE IF NECESSARY
#logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, 
#                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# User class for authentication
# V1
#class User(UserMixin):
#    def __init__(self, id):
#        self.id = id

# V2
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

    # Flask-Login requires this method to uniquely identify the user
    def get_id(self):
        return self.username

    @staticmethod
    def create_user(username, email, password):
        # Add a new user to DynamoDB
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

# Login route
#@app.route('/login', methods=['GET', 'POST'])
#def login():
#    if request.method == 'POST':
#        # Handle login logic here (username/password or Google login)
#        username = request.form['username']
#        # Simulate login (you should verify the username/password)
#        user = User(username)
#        login_user(user)
#        return redirect('/chat')
#    return render_template('login.html')
    
@app.route('/login', methods=['GET','POST'])
def login():
    
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
    
        # Fetch the user from DynamoDB
        user_data = User.get_user_by_id(username)
    
        if user_data and bcrypt.check_password_hash(user_data.password, password):
            # Login the user
            user = User(username=user_data.username, email=user_data.email, password=user_data.password)
            login_user(user)
            flash('Login successful!', 'success')
            return redirect(url_for('chat'))
        else:
            flash('Login failed. Please check your credentials.', 'danger')
            return redirect(url_for('login'))
    # If it's a GET request, render the registration form
    return render_template('login.html')  # Ensure 'login.html' exists in the templates folder

@app.route('/register', methods=['GET','POST'])
def register():
    
    if request.method == 'POST':
        # Handle the POST form submission
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
    
        # Hash the password
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
    
        # Check if user already exists
        if User.get_user_by_id(username):
            flash('Username already exists!', 'danger')
            return redirect(url_for('register'))
    
        # Create new user in DynamoDB
        User.create_user(username, email, hashed_password)
        flash('Registration successful!', 'success')
        return redirect(url_for('login'))
    
    # If it's a GET request, render the registration form
    return render_template('register.html')  # Ensure 'register.html' exists in the templates folder


# Chat page with WebSocket connection
@app.route('/chat')
#@login_required
def chat():
    return render_template('chat.html')

@app.route('/log_chat', methods=['POST'])
#@login_required
def log_chat():
    
    username = 'Guest'
    if current_user.is_authenticated:
        username = current_user.username
        
    # Get the JSON data from the POST request
    data = request.get_json()
    message = data.get('message')

    # Create a timestamp
    timestamp = datetime.now().isoformat()

    # Log chat message to DynamoDB
    chat_table.put_item(
        Item={
            'username': username,
            'timestamp': timestamp,
            'message': message
        }
    )

    # Return a success response
    return jsonify({'status': 'success', 'message': 'Message logged'}), 200


# WebSocket message handling
@socketio.on('message')
def handle_message(message):
    # Integrate your robot logic here
    #response = your_robot_script.process_message(message)
    logger.info(f"Message received: {message}")
    response = "Ok, I'll work on it buddy !"
    
    # Generate a unique ID for each message (you can use UUID or similar)
    message_id = str(uuid4())
    
    # Get the current timestamp
    timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')

    # Store the chat history in DynamoDB
    try:
        chat_table.put_item(
            Item={
                'message_id': message_id,           # Primary key
                'timestamp': timestamp,             # When the message was received
                'user_message': message,            # The message sent by the user
                'server_response': response,        # The server's response
            }
        )
        logger.info(f"Chat logged successfully: {message_id}")
    except Exception as e:
        logger.error(f"Error logging chat to DynamoDB: {e}")

    # Send the server response to the client
    send(response, broadcast=True)

# Log out route
@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect('/login')

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)