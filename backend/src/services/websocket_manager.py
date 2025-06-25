from flask_socketio import SocketIO, emit, join_room, leave_room, disconnect
from flask import request # Import request from flask

class WebSocketManager:
    def __init__(self, socketio):
        self.socketio = socketio
        self.connected_users = {}

    def send_message(self, user_id, event, data):
        """Send a message to a specific user"""
        if user_id in self.connected_users:
            self.socketio.emit(event, data, room=self.connected_users[user_id])

    def broadcast_message(self, event, data):
        """Broadcast a message to all connected users"""
        self.socketio.emit(event, data)

    def join_user_room(self, user_id, sid):
        """Add a user to their specific room"""
        self.connected_users[user_id] = sid
        join_room(sid)

    def leave_user_room(self, user_id):
        """Remove a user from their specific room"""
        if user_id in self.connected_users:
            leave_room(self.connected_users[user_id])
            del self.connected_users[user_id]

    def handle_connect(self, user_id):
        """Handle new WebSocket connection"""
        self.join_user_room(user_id, request.sid)
        print(f"User {user_id} connected. SID: {request.sid}")
        self.send_message(user_id, "status", {"message": "Connected to WebSocket"})

    def handle_disconnect(self, user_id):
        """Handle WebSocket disconnection"""
        self.leave_user_room(user_id)
        print(f"User {user_id} disconnected. SID: {request.sid}")

    def handle_custom_event(self, user_id, data):
        """Handle a custom event from the client"""
        print(f"Received custom event from {user_id}: {data}")
        self.send_message(user_id, "response", {"message": f"Received your message: {data}"})

websocket_manager = None

def get_websocket_manager(socketio=None):
    global websocket_manager
    if websocket_manager is None and socketio is not None:
        websocket_manager = WebSocketManager(socketio)
    return websocket_manager


