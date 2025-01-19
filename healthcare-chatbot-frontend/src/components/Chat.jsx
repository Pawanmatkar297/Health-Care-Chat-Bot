import React, { useState, useEffect, useRef } from 'react';
import { FaHistory, FaRobot, FaUser, FaSignOutAlt, FaPaperPlane, FaMicrophone, FaMicrophoneSlash } from 'react-icons/fa';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import './Chat.css';

const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
const recognition = new SpeechRecognition();

const Chat = () => {
    const [messages, setMessages] = useState([]);
    const [inputMessage, setInputMessage] = useState('');
    const [isListening, setIsListening] = useState(false);
    const [isWaitingForResponse, setIsWaitingForResponse] = useState(false);
    const [microphonePermission, setMicrophonePermission] = useState(false);
    const [isRecording, setIsRecording] = useState(false);
    const [mediaRecorder, setMediaRecorder] = useState(null);
    const [audioChunks, setAudioChunks] = useState([]);
    const navigate = useNavigate();
    const messagesEndRef = useRef(null);
    const sessionId = useRef(Math.random().toString(36).substring(7));

    useEffect(() => {
        // Add initial bot message when component mounts
        setMessages([{
            type: 'bot',
            content: 'Hello! How can I assist you today? Please describe your symptoms.',
            timestamp: new Date().toISOString()
        }]);

        // Check microphone permission
        checkMicrophonePermission();
    }, []);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    const sendMessageToBackend = async (message, type = 'text', audioBlob = null) => {
        try {
            setIsWaitingForResponse(true);
            let data;
            let headers = {};
            
            if (type === 'voice' && audioBlob) {
                data = new FormData();
                data.append('type', type);
                data.append('session_id', sessionId.current);
                data.append('audio', audioBlob, 'recording.wav');
                headers['Content-Type'] = 'multipart/form-data';
            } else {
                data = {
                    message: message,
                    type: type,
                    session_id: sessionId.current
                };
                headers['Content-Type'] = 'application/json';
            }

            const response = await axios.post('http://localhost:5002/api/chat', 
                type === 'voice' ? data : JSON.stringify(data),
                { 
                    headers,
                    timeout: 30000 // Increased timeout for voice processing
                }
            );

            console.log('Backend response:', response.data);

            if (response.data.success) {
                if (type === 'voice' && response.data.recognized_text) {
                    setMessages(prev => [...prev, {
                        type: 'user',
                        content: response.data.recognized_text,
                        timestamp: new Date().toISOString()
                    }]);
                } else if (type === 'text') {
                    setMessages(prev => [...prev, {
                        type: 'user',
                        content: message,
                        timestamp: new Date().toISOString()
                    }]);
                }

                setMessages(prev => [...prev, {
                    type: 'bot',
                    content: response.data.message,
                    timestamp: new Date().toISOString()
                }]);

                if (response.data.is_final) {
                    sessionId.current = Math.random().toString(36).substring(7);
                }
            } else {
                throw new Error(response.data.message || 'Error processing request');
            }
        } catch (error) {
            console.error('Error:', error);
            setMessages(prev => [...prev, {
                type: 'system',
                content: error.message || 'Error processing your message. Please try again.',
                timestamp: new Date().toISOString()
            }]);
        } finally {
            setIsWaitingForResponse(false);
        }
    };

    const handleSendMessage = async (e) => {
        e.preventDefault();
        if (inputMessage.trim() && !isWaitingForResponse) {
            const message = inputMessage.trim();
            setInputMessage('');
            await sendMessageToBackend(message, 'text');
        }
    };

    const checkMicrophonePermission = async () => {
        try {
            // Request microphone access
            const stream = await navigator.mediaDevices.getUserMedia({ 
                audio: true,
                video: false // explicitly disable video
            });
            
            // Stop all tracks after getting permission
            stream.getTracks().forEach(track => track.stop());
            
            setMicrophonePermission(true);
            console.log('Microphone permission granted');
        } catch (err) {
            console.error('Microphone permission error:', err);
            setMicrophonePermission(false);
            setMessages(prev => [...prev, {
                type: 'system',
                content: 'Please allow microphone access to use voice input.',
                timestamp: new Date().toISOString()
            }]);
        }
    };

    const handleRecording = async () => {
        if (!isRecording) {
            try {
                // Configure recognition
                recognition.continuous = false;
                recognition.lang = 'en-US';
                recognition.interimResults = false;
                recognition.maxAlternatives = 1;

                // Add event listeners
                recognition.onresult = (event) => {
                    const transcript = event.results[0][0].transcript;
                    console.log('Recognized text:', transcript);
                    
                    // Send the transcript to backend
                    sendMessageToBackend(transcript, 'text');
                };

                recognition.onerror = (event) => {
                    console.error('Speech recognition error:', event.error);
                    setMessages(prev => [...prev, {
                        type: 'system',
                        content: 'Error with voice recognition. Please try again.',
                        timestamp: new Date().toISOString()
                    }]);
                    setIsRecording(false);
                    setIsListening(false);
                };

                recognition.onend = () => {
                    setIsRecording(false);
                    setIsListening(false);
                    console.log('Speech recognition ended');
                };

                // Start recording
                recognition.start();
                setIsRecording(true);
                setIsListening(true);
                
                setMessages(prev => [...prev, {
                    type: 'system',
                    content: 'Listening... Speak now.',
                    timestamp: new Date().toISOString()
                }]);

            } catch (error) {
                console.error('Error starting recording:', error);
                setMessages(prev => [...prev, {
                    type: 'system',
                    content: 'Error accessing microphone. Please check permissions.',
                    timestamp: new Date().toISOString()
                }]);
                setIsRecording(false);
                setIsListening(false);
            }
        } else {
            try {
                recognition.stop();
                setIsRecording(false);
                setIsListening(false);
            } catch (error) {
                console.error('Error stopping recording:', error);
            }
        }
    };

    const toggleMicrophone = async () => {
        if (!isWaitingForResponse) {
            try {
                if (!microphonePermission) {
                    await checkMicrophonePermission();
                    if (!microphonePermission) {
                        return;
                    }
                }
                await handleRecording();
            } catch (error) {
                console.error('Microphone error:', error);
                setMessages(prev => [...prev, {
                    type: 'system',
                    content: 'Error with voice recording. Please try again or use text input.',
                    timestamp: new Date().toISOString()
                }]);
            }
        }
    };

    const handleLogout = () => {
        localStorage.removeItem('token');
        localStorage.removeItem('username');
        navigate('/login');
    };

    const renderMicButton = () => (
        <button 
            type="button" 
            className={`mic-button ${isRecording ? 'listening' : ''} ${!microphonePermission ? 'disabled' : ''}`}
            onClick={microphonePermission ? toggleMicrophone : checkMicrophonePermission}
            title={!microphonePermission ? 'Click to enable microphone access' : isRecording ? 'Click to stop recording' : 'Click to start recording'}
        >
            {isRecording ? <FaMicrophoneSlash /> : <FaMicrophone />}
        </button>
    );

    return (
        <div className="chat-container">
            {/* Sidebar */}
            <div className="sidebar">
                <div className="sidebar-header">
                    <FaRobot className="logo" />
                    <h2>Health Bot</h2>
                </div>
                
                <div className="sidebar-menu">
                    <button className="menu-item active">
                        <FaHistory /> Chat History
                    </button>
                    {/* <button className="menu-item">
                        <FaBook /> Prompt Library
                    </button> */}
                    {/* <button className="menu-item">
                        <FaStar /> Favorites
                    </button> */}
                    {/* <button className="menu-item">
                        <FaChartBar /> Statistics
                    </button> */}
                    {/* <button className="menu-item">
                        <FaCog /> Settings
                    </button> */}
                </div>

                <button className="logout-button" onClick={handleLogout}>
                    <FaSignOutAlt /> Log Out
                </button>
            </div>

            {/* Main Chat Area */}
            <div className="main-chat">
                <div className="chat-header">
                    <h2>Healthcare Chatbot</h2>
                    <div className="header-actions">
                        <button className="new-chat" onClick={() => {
                            sessionId.current = Math.random().toString(36).substring(7);
                            setMessages([{
                                type: 'bot',
                                content: 'Hello! How can I assist you today? Please describe your symptoms.',
                                timestamp: new Date().toISOString()
                            }]);
                        }}>+ New Chat</button>
                    </div>
                </div>

                <div className="messages-container">
                    {messages.map((message, index) => (
                        <div key={index} className={`message ${message.type}`}>
                            <div className="message-icon">
                                {message.type === 'user' ? <FaUser /> : <FaRobot />}
                            </div>
                            <div className="message-content">
                                {message.content}
                            </div>
                        </div>
                    ))}
                    <div ref={messagesEndRef} />
                </div>

                <form className={`input-area ${isWaitingForResponse ? 'disabled' : ''}`} onSubmit={handleSendMessage}>
                    <input
                        type="text"
                        value={inputMessage}
                        onChange={(e) => setInputMessage(e.target.value)}
                        placeholder="Type your symptoms here..."
                        disabled={isWaitingForResponse}
                    />
                    {renderMicButton()}
                    <button type="submit" disabled={isWaitingForResponse}>
                        <FaPaperPlane />
                    </button>
                </form>
            </div>
        </div>
    );
};

export default Chat; 