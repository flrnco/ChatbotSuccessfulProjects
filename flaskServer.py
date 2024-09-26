from flask import Flask, render_template, request
from flask_socketio import SocketIO, send
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user
#import your_robot_script  # This is where your robot's Python logic is handled


   
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
socketio = SocketIO(app)

# User login management
login_manager = LoginManager()
login_manager.init_app(app)

# User class for authentication
class User(UserMixin):
    def __init__(self, id):
        self.id = id

@login_manager.user_loader
def load_user(user_id):
    return User(user_id)

@app.route('/')
def index():
    return render_template('index.html')

# Login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Handle login logic here (username/password or Google login)
        username = request.form['username']
        # Simulate login (you should verify the username/password)
        user = User(username)
        login_user(user)
        return redirect('/chat')
    return render_template('login.html')

# Chat page with WebSocket connection
@app.route('/chat')
#@login_required
def chat():
    return render_template('chat.html')

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