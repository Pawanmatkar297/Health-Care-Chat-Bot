import React, { useState, useEffect, useRef } from 'react';
import { FaHistory, FaRobot, FaUser, FaSignOutAlt, FaPaperPlane, FaMicrophone, FaMicrophoneSlash, FaMoon, FaBars, FaChevronLeft, FaUserMd } from 'react-icons/fa';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import './Chat.css';

const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
const recognition = SpeechRecognition ? new SpeechRecognition() : null;

const Chat = () => {
    const [messages, setMessages] = useState([]);
    const [inputMessage, setInputMessage] = useState('');
    const [isListening, setIsListening] = useState(false);
    const [isWaitingForResponse, setIsWaitingForResponse] = useState(false);
    const [microphonePermission, setMicrophonePermission] = useState(false);
    const [isRecording, setIsRecording] = useState(false);
    const [mediaRecorder, setMediaRecorder] = useState(null);
    const [audioChunks, setAudioChunks] = useState([]);
    const [isStarted, setIsStarted] = useState(false);
    const [darkMode, setDarkMode] = useState(() => {
        const savedMode = localStorage.getItem('darkMode');
        return savedMode ? JSON.parse(savedMode) : false;
    });
    const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false);
    const [currentTypingMessage, setCurrentTypingMessage] = useState('');
    const [isTyping, setIsTyping] = useState(false);
    const navigate = useNavigate();
    const messagesEndRef = useRef(null);
    const sessionId = useRef(Math.random().toString(36).substring(7));

    useEffect(() => {
        if (darkMode) {
            document.body.classList.add('dark-mode');
        } else {
            document.body.classList.remove('dark-mode');
        }
        localStorage.setItem('darkMode', JSON.stringify(darkMode));
    }, [darkMode]);

    useEffect(() => {
        // Only add initial bot message when component mounts if not using get started button
        if (isStarted) {
            setMessages([{
                type: 'bot',
                content: 'Hello! How can I assist you today? Please describe your symptoms.',
                timestamp: new Date().toISOString()
            }]);
        }

        // Check microphone permission
        checkMicrophonePermission();
    }, [isStarted]);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    // Text to Speech setup
    const speak = (text) => {
        if ('speechSynthesis' in window) {
            const utterance = new SpeechSynthesisUtterance(text);
            utterance.rate = 1; // Speed of speech
            utterance.pitch = 1; // Pitch of voice
            utterance.volume = 1; // Volume
            window.speechSynthesis.speak(utterance);
        }
    };

    // Typing animation function
    const typeMessage = async (message) => {
        setIsTyping(true);
        let currentText = '';
        const delay = 30; // Delay between each character

        // Add temporary typing message
        setMessages(prev => [...prev, {
            type: 'bot',
            content: '',
            isTyping: true,
            timestamp: new Date().toISOString()
        }]);

        for (let i = 0; i < message.length; i++) {
            currentText += message[i];
            setMessages(prev => prev.map((msg, index) => {
                if (index === prev.length - 1 && msg.isTyping) {
                    return { ...msg, content: currentText };
                }
                return msg;
            }));
            await new Promise(resolve => setTimeout(resolve, delay));
        }

        // Update the final message and speak
        setMessages(prev => prev.map((msg, index) => {
            if (index === prev.length - 1 && msg.isTyping) {
                return {
                    type: 'bot',
                    content: message,
                    timestamp: new Date().toISOString()
                };
            }
            return msg;
        }));
        
        setIsTyping(false);
        speak(message); // Speak after typing is complete
    };

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
                    timeout: 30000,
                    withCredentials: true
                }
            );

            if (response.data.success) {
                // Only add user message for voice input
                if (type === 'voice' && response.data.recognized_text) {
                    setMessages(prev => [...prev, {
                        type: 'user',
                        content: response.data.recognized_text,
                        timestamp: new Date().toISOString()
                    }]);
                }

                // Add bot's response with typing animation
                await typeMessage(response.data.message);

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
            setInputMessage(''); // Clear input immediately
            
            // Add user message first
            setMessages(prev => [...prev, {
                type: 'user',
                content: message,
                timestamp: new Date().toISOString()
            }]);

            // Then send to backend
            await sendMessageToBackend(message, 'text');
            await saveChatHistory(); // Save after each message
        }
    };

    const saveChatHistory = async () => {
        try {
            const token = localStorage.getItem('token');
            if (!token) return;

            await axios.post('http://localhost:5001/api/chat-history/save', {
                sessionId: sessionId.current,
                messages: messages
            }, {
                headers: { Authorization: `Bearer ${token}` }
            });
        } catch (error) {
            console.error('Error saving chat history:', error);
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
        if (!recognition) {
            setMessages(prev => [...prev, {
                type: 'system',
                content: 'Speech recognition is not supported in your browser. Please use Chrome.',
                timestamp: new Date().toISOString()
            }]);
            return;
        }

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
                    
                    // Add the transcript to messages
                    setMessages(prev => [...prev, {
                        type: 'user',
                        content: transcript,
                        timestamp: new Date().toISOString()
                    }]);
                    
                    // Send the transcript to backend
                    sendMessageToBackend(transcript, 'text');
                };

                recognition.onerror = (event) => {
                    console.error('Speech recognition error:', event.error);
                    let errorMessage = 'Error with voice recognition. ';
                    
                    switch(event.error) {
                        case 'network':
                            errorMessage += 'Please check your internet connection.';
                            break;
                        case 'not-allowed':
                            errorMessage += 'Please allow microphone access in your browser settings.';
                            break;
                        case 'no-speech':
                            errorMessage += 'No speech was detected. Please try again.';
                            break;
                        default:
                            errorMessage += 'Please try again.';
                    }
                    
                    setMessages(prev => [...prev, {
                        type: 'system',
                        content: errorMessage,
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
                await recognition.start();
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
                    content: 'Error accessing microphone. Please check browser permissions and try again.',
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

    const handleGetStarted = () => {
        setIsStarted(true);
    };

    const viewChatHistory = () => {
        navigate('/chat-history');
    };

    const toggleDarkMode = () => {
        setDarkMode(prev => !prev);
    };

    const toggleSidebar = () => {
        setIsSidebarCollapsed(!isSidebarCollapsed);
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

    const renderWelcomeScreen = () => (
        <div className="get-started-container">
            <div className="welcome-icon">
                <FaUserMd size={48} color="var(--primary-color)" />
            </div>
            <h2>Welcome to MedAssist AI</h2>
            <p>Your personal healthcare assistant powered by artificial intelligence</p>
            <div className="feature-grid">
                <div className="feature-item">
                    <FaRobot size={24} />
                    <h3>AI-Powered Analysis</h3>
                    <p>Advanced symptom analysis using machine learning</p>
                </div>
                <div className="feature-item">
                    <FaMicrophone size={24} />
                    <h3>Voice Enabled</h3>
                    <p>Speak your symptoms naturally</p>
                </div>
                <div className="feature-item">
                    <FaHistory size={24} />
                    <h3>Consultation History</h3>
                    <p>Track your health conversations</p>
                </div>
            </div>
            <button className="get-started-button" onClick={handleGetStarted}>
                Start Consultation
            </button>
            <p className="disclaimer">
                Note: This is not a replacement for professional medical advice. 
                Always consult with a healthcare provider for medical decisions.
            </p>
        </div>
    );

    const renderMessage = (message, index) => {
        const isSystem = message.type === 'system';
        const isUser = message.type === 'user';
        const icon = isUser ? <FaUser /> : isSystem ? <FaRobot color="#FFC107" /> : <FaRobot />;
        
        return (
            <div key={index} className={`message ${message.type} ${message.isTyping ? 'typing' : ''}`}>
                <div className="message-icon">
                    {icon}
                </div>
                <div className="message-content">
                    {message.content}
                    {message.isTyping && <span className="typing-cursor"/>}
                </div>
            </div>
        );
    };

    const renderSidebar = () => (
        <div className={`sidebar ${isSidebarCollapsed ? 'collapsed' : ''}`}>
            <div className="sidebar-header">
                <img src="/bot_assistant_QWK_icon.ico" alt="MedAssist Logo" className="clogo" />
                {!isSidebarCollapsed && <h2>MedAssist</h2>}
                <button className="collapse-button" onClick={toggleSidebar}>
                    {isSidebarCollapsed ? <FaBars /> : <FaChevronLeft />}
                </button>
            </div>
            
            <div className="sidebar-menu">
                <div className="menu-section">
                    <button className="menu-item active" onClick={viewChatHistory}>
                        <FaHistory /> {!isSidebarCollapsed && 'Consultation History'}
                    </button>
                </div>

                <div className="menu-section settings">
                    <h3 className="menu-title">{!isSidebarCollapsed && 'Preferences'}</h3>
                    <label className="toggle-switch">
                        <span>
                            <FaMoon /> {!isSidebarCollapsed && 'Dark Mode'}
                        </span>
                        <div className="switch">
                            <input
                                type="checkbox"
                                checked={darkMode}
                                onChange={toggleDarkMode}
                            />
                            <span className="slider"></span>
                        </div>
                    </label>
                </div>
            </div>

            <div className="sidebar-footer">
                {!isSidebarCollapsed && (
                    <div className="user-info">
                        <div className="user-avatar">
                            <FaUser />
                        </div>
                        <span className="username">{localStorage.getItem('username')}</span>
                    </div>
                )}
                <button className="logout-button" onClick={handleLogout}>
                    <FaSignOutAlt /> {!isSidebarCollapsed && 'End Session'}
                </button>
            </div>
        </div>
    );

    return (
        <div className="chat-container">
            {renderSidebar()}
            
            <div className="main-chat">
                <div className="chat-header">
                    <div className="header-info">
                        <h2>Hello {localStorage.getItem('username')}</h2>
                        <p className="consultation-id">Consultation ID: {sessionId.current}</p>
                    </div>
                    <div className="header-actions">
                        <button 
                            className="new-chat" 
                            onClick={() => {
                                sessionId.current = Math.random().toString(36).substring(7);
                                setMessages([{
                                    type: 'bot',
                                    content: 'Hello! I\'m here to help. Please describe your symptoms in detail.',
                                    timestamp: new Date().toISOString()
                                }]);
                            }}
                        >
                            + New Consultation
                        </button>
                    </div>
                </div>

                <div className="messages-container">
                    {!isStarted ? renderWelcomeScreen() : (
                        <>
                            {messages.map((message, index) => renderMessage(message, index))}
                            {isWaitingForResponse && (
                                <div className="message bot">
                                    <div className="message-icon">
                                        <FaRobot />
                                    </div>
                                    <div className="message-content typing">
                                        <div className="typing-indicator">
                                            <span></span>
                                            <span></span>
                                            <span></span>
                                        </div>
                                    </div>
                                </div>
                            )}
                        </>
                    )}
                    <div ref={messagesEndRef} />
                </div>

                <form 
                    className={`input-area ${isWaitingForResponse || !isStarted ? 'disabled' : ''}`} 
                    onSubmit={handleSendMessage}
                >
                    <input
                        type="text"
                        value={inputMessage}
                        onChange={(e) => setInputMessage(e.target.value)}
                        placeholder="Describe your symptoms here..."
                        disabled={isWaitingForResponse || !isStarted}
                    />
                    {renderMicButton()}
                    <button 
                        type="submit" 
                        disabled={isWaitingForResponse || !isStarted || !inputMessage.trim()}
                    >
                        <FaPaperPlane />
                    </button>
                </form>
            </div>
        </div>
    );
};

export default Chat; 