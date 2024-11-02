import socketio

# Initialize SocketIO client
sio = socketio.Client()

@sio.event
def connect():
    print('Connected to server')

@sio.event
def disconnect():
    print('Disconnected from server')

@sio.on('chat_response')
def on_chat_response(data):
    print('Received response:', data)
    # Optionally, disconnect after receiving the response if you don't need to keep the client running
    sio.disconnect()

@sio.on('error')
def on_error(data):
    print('Error:', data)

@sio.on('lead_form_response')
def on_lead_form_response(data):
    print('Lead form response:', data)

@sio.on('user_id')
def on_user_id(data):
    print('Received user_id:', data)
    # Store the user_id for future messages
    sio.user_id = data['user_id']

def main():
    sio.connect('http://localhost:8080')
    # Wait for connection
    sio.sleep(1)

    message = "Can I get special conditions as a start-up?"
    data = {
        'message': message,
        'user_id': 1, # None 
        'InitialPrompt': '0'
    }
    sio.emit('chat_message', data)

    # Wait indefinitely for events
    sio.wait()

if __name__ == '__main__':
    main()
