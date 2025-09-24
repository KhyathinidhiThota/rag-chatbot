# Simple in-memory multi-turn session handling
from collections import defaultdict

class MemoryStore:
    def __init__(self):
        self.sessions = defaultdict(list)

    def add_turn(self, session_id, user, assistant):
        self.sessions[session_id].append({"user": user, "assistant": assistant})

    def get_history(self, session_id):
        return self.sessions[session_id]

memory_store = MemoryStore()
