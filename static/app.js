const { useState, useEffect, useRef, useCallback } = React;

// useCallback to prevent unnecessary re-creations on each render

// API Service for handling chat-related requests
const apiService = {
    async sendChatMessage(message, session_id) {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message, session_id }),
        });
        if (!response.ok) throw new Error('Failed to get response from API');
        return response.json();
    },

    async refreshIndex() {
        const response = await fetch('/api/refresh-index', { method: 'POST' });
        if (!response.ok) throw new Error('Failed to get response from API');
        return response.json();
    },

    async getHistory(sessionId) {
        const response = await fetch(`/api/get-history?session_id=${sessionId}`);
        if (!response.ok) throw new Error('Failed to fetch chat history');
        return response.json();
    },
};

/**
 * Notification Component
 * Displays temporary success or error messages to the user.
 * Automatically dismisses after 3 seconds.
 */
const Notification = ({ message, type, onClose }) => {
    useEffect(() => { 
        const timer = setTimeout(onClose, 3000); // Auto-close notification
        lucide.createIcons();
        return () => clearTimeout(timer); // Cleanup timer on unmount
    }, [onClose]);

    // Determine background color and icon based on notification type
    const bgColor = type === 'error' ? 'bg-red-500' : 'bg-green-500';
    const icon = type === 'error' ? 'alert-circle' : 'check-circle';

    return (
        <div className={`${bgColor} text-white text-sm px-4 py-2 rounded-lg shadow-lg fade-in flex items-center`}>
            <i data-lucide={icon} className="w-4 h-4 mr-2"></i>
            {message}
        </div>
    );
};

/**
 * Chat Component
 * Root component managing the chat interface, including message handling, API interactions,
 * and integrating other sub-components like Header, ChatWindow, Actions, and Footer.
 */
const Chat = () => {
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [isRefreshing, setIsRefreshing] = useState(false);
    const [notification, setNotification] = useState(null);
    const [sessionId, setSessionId] = useState(() => localStorage.getItem('session_id') || generateID());

    const messagesEndRef = useRef(null);
    const inputRef = useRef(null);

    // Persist session ID
    useEffect(() => {
        localStorage.setItem('session_id', sessionId);
    }, [sessionId]);

    // Scroll to bottom helper
    const scrollToBottom = useCallback(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, []);

    // Fetch chat history on mount
    useEffect(() => {
        const fetchHistory = async () => {
            try {
                const history = await apiService.getHistory(sessionId);
                const formattedHistory = history.map(([text, isUser]) => ({ text, isUser: Boolean(isUser) }));
                setMessages(formattedHistory);
                scrollToBottom();
            } catch (error) {
                console.error('Error fetching chat history:', error);
            } finally {
                lucide.createIcons();
            }
        };

        fetchHistory();
    }, [sessionId, scrollToBottom]);

    // Auto-focus input on mount
    useEffect(() => {
        inputRef.current?.focus();
    }, []);

    // Handler to send messages
    const sendMessage = useCallback(
        async (e) => {
            e.preventDefault();
            const trimmedInput = input.trim();
            if (!trimmedInput) return; // Prevent sending empty messages

            setIsLoading(true);
            setMessages((prev) => [...prev, { text: trimmedInput, isUser: true }]);
            setInput('');

            try {
                const data = await apiService.sendChatMessage(trimmedInput, sessionId);
                setMessages((prev) => [...prev, { text: data.response, isUser: false }]);
            } catch (error) {
                setMessages((prev) => [...prev, { text: error.message, isUser: false }]);
            } finally {
                setIsLoading(false);
                inputRef.current?.focus(); // Refocus input after sending
                scrollToBottom();
            }
        },
        [input, sessionId, scrollToBottom]
    );

    // Handler to refresh index
    const refreshIndex = useCallback(async () => {
        setIsRefreshing(true);
        try {
            const response = await apiService.refreshIndex();
            setNotification({ message: response.message, type: 'success' });
        } catch (error) {
            setNotification({ message: error.message, type: 'error' });
        } finally {
            setIsRefreshing(false);
        }
    }, []);

    // Handler to reset chat
    const resetChat = useCallback(() => {
        setSessionId(generateID());
        setMessages([]);
        setInput('');
        setNotification({ message: 'Chat history cleared.', type: 'success' });
    }, []);

    return (
        <div className="md:w-[600px] w-full mx-auto bg-white rounded-xl shadow-2xl overflow-hidden">
            <Header />
            <ChatWindow
                messages={messages}
                isLoading={isLoading}
                sendMessage={sendMessage}
                input={input}
                setInput={setInput}
                inputRef={inputRef}
                messagesEndRef={messagesEndRef}
            />
            <Actions
                refreshIndex={refreshIndex}
                resetChat={resetChat}
                isRefreshing={isRefreshing}
                notification={notification}
                setNotification={setNotification}
            />
            <Footer />
        </div>
    );
};

/**
 * Header Component
 * Displays the application's header with title and attribution.
 */
const Header = () => (
    <header className="bg-gradient-to-r from-indigo-600 to-indigo-800 text-white p-6 py-4">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between">
            <h1 className="text-xl font-bold">Demo: Bank Customer Support</h1>
            <div className="flex items-center text-sm space-x-2 mt-2 sm:mt-0 justify-end pl-8">
                <span>
                    Powered by
                    <a href="https://github.com/langchain-ai/langchain" target="_blank" className="underline pl-1">
                        LangChain
                    </a>
                </span>
                🦜️🔗
            </div>
        </div>
    </header>
);

/**
 * ChatWindow Component
 * Renders the chat messages, input field, and send button.
 * Handles the display of loading indicators and message styling.
 * 
 * React.memo to avoid re-rendering unless its props change.
 */
const ChatWindow = React.memo(
    ({ messages, isLoading, sendMessage, input, setInput, inputRef, messagesEndRef }) => (
        <div className="p-6 h-[calc(100vh-240px)] min-h-[300px] flex flex-col">
            <div className="bg-gray-50 rounded-lg border border-gray-200 shadow-inner overflow-hidden flex-1">
            <div className="h-full overflow-y-auto p-4 space-y-4" style={{ scrollBehavior: 'smooth' }}>
                    {messages.length === 0 && (
                        <div className="text-center text-gray-500 mt-4">
                            Start a conversation by sending a message.
                        </div>
                    )}
                    {messages.map((msg, index) => (
                        <div key={index} className={`flex ${msg.isUser ? 'justify-end' : 'justify-start'}`}>
                            <div
                                className={`max-w-xs lg:max-w-md px-4 py-2 rounded-lg shadow-md ${
                                    msg.isUser
                                        ? 'bg-gradient-to-r from-indigo-500 to-indigo-600 text-white ml-auto'
                                        : 'bg-white border border-gray-200'
                                } fade-in`}
                            >
                                {msg.text.split("\n").map((line, index) => (
                                    <span key={index}>
                                        {line}
                                        <br />
                                    </span>
                                ))}
                            </div>
                        </div>
                    ))}
                    {isLoading && (
                        <div className="flex justify-start">
                            <div className="bg-gray-200 text-gray-700 px-4 py-2 rounded-lg typing-indicator">
                                AI is thinking
                            </div>
                        </div>
                    )}
                    <div ref={messagesEndRef} /> {/* Reference point to scroll into view */}
                </div>
            </div>
            <form onSubmit={sendMessage} className="pt-6 bg-white rounded-b-lg">
                <div className="flex border border-gray-200 rounded-lg overflow-hidden transition-all duration-200">
                    <input
                        ref={inputRef}
                        type="text"
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        placeholder="Ask a question..."
                        className="flex-grow px-4 py-2 focus:outline-none"
                        disabled={isLoading} // Disable while loading
                    />
                    <button
                        type="submit"
                        className="bg-indigo-500 text-white px-6 py-2 hover:bg-indigo-600 disabled:opacity-50 transition-all duration-200 flex items-center"
                        disabled={isLoading} // Disable while loading
                    >
                        <span>{isLoading ? 'Sending...' : 'Send'}</span>
                        <i data-lucide={isLoading ? 'loader' : 'send'} className="w-4 h-4 ml-2"></i>
                    </button>
                </div>
            </form>
        </div>
    )
);

/**
 * Actions Component
 * Provides buttons for resetting the chat and refreshing the index.
 * Displays notifications based on user actions.
 */
const Actions = ({ refreshIndex, resetChat, isRefreshing, notification, setNotification }) => (
    <div className="p-6 pt-0 text-center relative">
        <div className="flex justify-center space-x-4">
            <button
                onClick={resetChat}
                className="bg-gradient-to-r from-red-100 to-red-200
                text-red-500 hover:text-white px-6 py-2 rounded-lg hover:from-red-400 hover:to-red-600 
                focus:outline-none focus:ring-2 focus:ring-red-500 focus:ring-offset-2 
                transition-all duration-200 shadow-md flex items-center"
            >
                <span className="text-sm">Reset Chat</span>
                <i data-lucide="trash-2" className="w-4 h-4 ml-2"></i>
            </button>
            <button
                onClick={refreshIndex}
                className={`bg-gradient-to-r ${
                    isRefreshing ? 'from-gray-400 to-gray-500' : 'from-green-500 to-green-600'
                    } text-white px-6 py-2 rounded-lg hover:from-green-600 hover:to-green-700 
                    focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-offset-2 
                    disabled:opacity-50 transition-all duration-200 shadow-md flex items-center mx-auto`}
                disabled={isRefreshing} // Disable while refreshing
            >
                <span className="text-sm">{isRefreshing ? 'Refreshing...' : 'Refresh Index'}</span>
                <i data-lucide="refresh-cw" className="w-4 h-4 ml-2"></i>
            </button>
        </div>
        {notification && (
            <div className="absolute left-1/2 transform -translate-x-1/2 mt-2">
                <Notification message={notification.message} type={notification.type} onClose={() => setNotification(null)} />
            </div>
        )}
    </div>
);

/**
 * Footer Component
 * Displays footer information with credits and links.
 */
const Footer = () => (
    <footer className="bg-gray-100 p-6 py-4 text-center text-sm text-gray-600 border-t border-gray-200">
        <p className="flex items-center justify-center">
            <span>Built by</span>
            <a href="https://multinear.com" target="_blank" className="text-indigo-600 hover:text-indigo-800 px-1">Multinear</a>
            2024 
            <span className="text-gray-400 px-2">|</span>
            <a href="https://github.com/multinear-demo/demo-bank-customer-support-py" 
                target="_blank" 
                className="text-indigo-600 hover:text-indigo-800 flex items-center">
                <span className="border border-gray-500 rounded-full p-1 mr-1">
                    <i data-lucide="github" className="w-3 h-3 text-gray-500"></i>
                </span>
                Repo
            </a>
            <span className="text-gray-400 px-2">|</span>
            <a href="https://multinear.com/docs/" 
                target="_blank" 
                className="text-indigo-600 hover:text-indigo-800 flex items-center">
                <span className="border border-gray-500 rounded-full p-1 mr-1">
                    <i data-lucide="book-open" className="w-3 h-3 text-gray-500"></i>
                </span>
                Docs
            </a>
        </p>
    </footer>
);

/**
 * Renders the Chat component into the root DOM element.
 * Entry point of the React application.
 */
ReactDOM.render(<Chat />, document.getElementById('root'));

// Utility function to generate ID
function generateID() {
    return window.crypto?.randomUUID?.() || Date.now().toString();
}
