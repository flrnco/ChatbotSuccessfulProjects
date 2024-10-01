from flask import Flask, render_template, request
from flask_socketio import SocketIO, send
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user
#import your_robot_script  # This is where your robot's Python logic is handled
import boto3
# libraries to manage credentials
from flask import request, redirect, url_for, flash
from flask_bcrypt import Bcrypt
import uuid

  
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
socketio = SocketIO(app)

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

# User class for authentication
# V1
#class User(UserMixin):
#    def __init__(self, id):
#        self.id = id

# V2
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User(user_id)

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
    
@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']

    # Fetch user details from DynamoDB
    response = users_table.get_item(
        Key={
            'username': username
        }
    )

    user = response.get('Item')
    
    if user and bcrypt.check_password_hash(user['password'], password):
        flash('Login successful!', 'success')
        # Add your logic for session handling
        return redirect(url_for('chat'))
    else:
        flash('Login failed. Check your username and password.', 'danger')
        return redirect(url_for('login'))

@app.route('/register', methods=['POST'])
def register():
    username = request.form['username']
    email = request.form['email']
    password = request.form['password']

    # Hash the password
    hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')

    # Insert into DynamoDB
    users_table.put_item(
        Item={
            'username': username,
            'email': email,
            'password': hashed_password
        }
    )

    flash('Registration successful!', 'success')
    return redirect(url_for('chat'))


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
    response = "Ok, I'll work on it buddy !"
    send(response, broadcast=True)

# Log out route
@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect('/login')

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)