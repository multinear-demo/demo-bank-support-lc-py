from typing import Dict, List, Tuple


class SessionManager:
    """
    Manages session message history. Uses in-memory storage.
    In production, we recommend using Redis or another persistent storage.
    """

    def __init__(self):
        self.sessions: Dict[str, List[Tuple[str, bool]]] = {}

    def get_history(self, chat_id: str) -> List[Tuple[str, bool]]:
        """
        Retrieves the message history for a given chat.
        """
        return [tuple(msg) for msg in self.sessions.get(chat_id, [])]

    def add_message(self, chat_id: str, message: Tuple[str, bool]) -> None:
        """
        Adds a message to the session history.
        """
        if chat_id not in self.sessions:
            self.sessions[chat_id] = []
        self.sessions[chat_id].append(message)
