import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import { FaUser, FaRobot, FaArrowLeft, FaCalendar, FaClock } from 'react-icons/fa';
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

            const response = await axios.get(`${process.env.REACT_APP_AUTH_URL}/api/chat-history/sessions`, {
                headers: { 
                    Authorization: `Bearer ${token}`,
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                },
                withCredentials: true
            });

            if (response.data.success) {
                // Sort sessions by date, newest first
                const sortedSessions = response.data.chatHistories.sort((a, b) => 
                    new Date(b.createdAt) - new Date(a.createdAt)
                );
                setChatSessions(sortedSessions);
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
            const response = await axios.get(`${process.env.REACT_APP_AUTH_URL}/api/chat-history/session/${sessionId}`, {
                headers: { 
                    Authorization: `Bearer ${token}`,
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                },
                withCredentials: true
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
        const options = {
            year: 'numeric',
            month: 'long',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        };
        return new Date(dateString).toLocaleString(undefined, options);
    };

    const formatTime = (dateString) => {
        const options = {
            hour: '2-digit',
            minute: '2-digit'
        };
        return new Date(dateString).toLocaleString(undefined, options);
    };

    const handleBackToChat = () => {
        navigate('/chat');
    };

    const getFinalDiagnosis = (messages) => {
        // Find the last bot message after user says "no"
        let foundNo = false;
        let finalDiagnosis = '';
        
        for (let i = messages.length - 1; i >= 0; i--) {
            const message = messages[i];
            
            // Check for system message indicating completion
            if (message.type === 'system' && message.content.includes('Final assessment')) {
                continue; // Skip the completion message
            }
            
            if (message.type === 'user' && message.content.toLowerCase() === 'no') {
                foundNo = true;
            } else if (foundNo && message.type === 'bot') {
                // This is the final diagnosis message
                finalDiagnosis = message.content;
                break;
            }
        }
        
        return finalDiagnosis;
    };

    const renderMessages = (messages) => {
        return messages.map((message, index) => {
            // Skip the completion message in the chat display
            if (message.type === 'system' && message.content.includes('Final assessment')) {
                return null;
            }
            
            return (
                <div key={index} className={`message ${message.type}`}>
                    <div className="message-icon">
                        {message.type === 'user' ? <FaUser /> : <FaRobot />}
                    </div>
                    <div className="message-content">
                        <div className="message-text">{message.content}</div>
                        <div className="message-timestamp">
                            <FaClock /> {formatTime(message.timestamp)}
                        </div>
                    </div>
                </div>
            );
        });
    };

    if (loading) {
        return (
            <div className="chat-history-container loading">
                <div className="loading-spinner"></div>
                <p>Loading chat history...</p>
            </div>
        );
    }

    if (error) {
        return (
            <div className="chat-history-container error">
                <div className="error-message">
                    <p>{error}</p>
                    <button onClick={handleBackToChat}>Back to Chat</button>
                </div>
            </div>
        );
    }

    return (
        <div className="chat-history-container">
            <div className="chat-history-header">
                <button onClick={handleBackToChat} className="back-button">
                    <FaArrowLeft /> Back to Chat
                </button>
                <h2>Consultation History</h2>
            </div>

            <div className="chat-history-content">
                <div className="sessions-list">
                    <h3>Past Consultations</h3>
                    {chatSessions.length === 0 ? (
                        <div className="no-sessions">
                            <p>No consultation history found</p>
                        </div>
                    ) : (
                        chatSessions.map((session) => (
                            <div
                                key={session._id}
                                className={`session-item ${selectedSession?._id === session._id ? 'selected' : ''}`}
                                onClick={() => fetchSessionDetails(session.sessionId)}
                            >
                                <div className="session-header">
                                    <div className="session-date">
                                        <FaCalendar /> {formatDate(session.createdAt)}
                                    </div>
                                    <div className="session-id">ID: {session.sessionId}</div>
                                </div>
                                <div className="session-preview">
                                    {session.messages[0]?.content.substring(0, 100)}...
                                </div>
                                {session.messages.some(m => m.type === 'user' && m.content.toLowerCase() === 'no') && (
                                    <div className="session-status">
                                        Consultation Completed
                                    </div>
                                )}
                            </div>
                        ))
                    )}
                </div>

                <div className="session-details">
                    {selectedSession ? (
                        <div className="messages-container">
                            <div className="session-info">
                                <h3>Consultation Details</h3>
                                <p>Session ID: {selectedSession.sessionId}</p>
                                <p>Date: {formatDate(selectedSession.createdAt)}</p>
                                {selectedSession.messages.some(m => m.type === 'user' && m.content.toLowerCase() === 'no') && (
                                    <p className="session-complete">Status: Completed</p>
                                )}
                            </div>
                            
                            <div className="messages-list">
                                {renderMessages(selectedSession.messages)}
                            </div>

                            {selectedSession.messages.some(m => m.type === 'user' && m.content.toLowerCase() === 'no') && (
                                <div className="final-diagnosis">
                                    <h4>Final Assessment</h4>
                                    <p>{getFinalDiagnosis(selectedSession.messages)}</p>
                                </div>
                            )}
                        </div>
                    ) : (
                        <div className="no-session-selected">
                            <FaRobot className="large-icon" />
                            <p>Select a consultation from the list to view details</p>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

export default ChatHistory; 