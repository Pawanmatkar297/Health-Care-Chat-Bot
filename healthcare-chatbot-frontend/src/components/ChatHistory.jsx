import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import './ChatHistory.css';

const ChatHistory = () => {
    const [chatSessions, setChatSessions] = useState([]);
    const [selectedSession, setSelectedSession] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const navigate = useNavigate();

    useEffect(() => {
        fetchChatSessions();
    }, []);

    const fetchChatSessions = async () => {
        try {
            const token = localStorage.getItem('token');
            if (!token) {
                navigate('/login');
                return;
            }

            const response = await axios.get('http://localhost:5001/api/chat-history/sessions', {
                headers: { Authorization: `Bearer ${token}` }
            });

            if (response.data.success) {
                setChatSessions(response.data.chatHistories);
            }
        } catch (error) {
            console.error('Error fetching chat sessions:', error);
            setError('Failed to load chat history');
        } finally {
            setLoading(false);
        }
    };

    const fetchSessionDetails = async (sessionId) => {
        try {
            const token = localStorage.getItem('token');
            const response = await axios.get(`http://localhost:5001/api/chat-history/session/${sessionId}`, {
                headers: { Authorization: `Bearer ${token}` }
            });

            if (response.data.success) {
                setSelectedSession(response.data.chatHistory);
            }
        } catch (error) {
            console.error('Error fetching session details:', error);
            setError('Failed to load session details');
        }
    };

    const formatDate = (dateString) => {
        return new Date(dateString).toLocaleString();
    };

    const handleBackToChat = () => {
        navigate('/chat');
    };

    if (loading) {
        return <div className="chat-history-container">Loading...</div>;
    }

    if (error) {
        return <div className="chat-history-container">Error: {error}</div>;
    }

    return (
        <div className="chat-history-container">
            <div className="chat-history-header">
                <h2>Chat History</h2>
                <button onClick={handleBackToChat} className="back-button">
                    Back to Chat
                </button>
            </div>

            <div className="chat-history-content">
                <div className="sessions-list">
                    {chatSessions.map((session) => (
                        <div
                            key={session._id}
                            className={`session-item ${selectedSession?._id === session._id ? 'selected' : ''}`}
                            onClick={() => fetchSessionDetails(session.sessionId)}
                        >
                            <div className="session-date">{formatDate(session.createdAt)}</div>
                            <div className="session-preview">
                                {session.messages[0]?.content.substring(0, 50)}...
                            </div>
                        </div>
                    ))}
                </div>

                <div className="session-details">
                    {selectedSession ? (
                        <div className="messages-container">
                            <h3>Chat Session: {formatDate(selectedSession.createdAt)}</h3>
                            {selectedSession.messages.map((message, index) => (
                                <div key={index} className={`message ${message.type}`}>
                                    <div className="message-content">{message.content}</div>
                                    <div className="message-timestamp">
                                        {formatDate(message.timestamp)}
                                    </div>
                                </div>
                            ))}
                        </div>
                    ) : (
                        <div className="no-session-selected">
                            Select a chat session to view details
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

export default ChatHistory; 