import React, { useState, useEffect, useRef } from 'react';
import { FaHistory, FaRobot, FaUser, FaSignOutAlt, FaPaperPlane, FaMicrophone, FaMicrophoneSlash, FaMoon, FaBars, FaChevronLeft, FaUserMd, FaLanguage } from 'react-icons/fa';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import './Chat.css';

const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
const recognition = SpeechRecognition ? new SpeechRecognition() : null;

const Chat = () => {
    const [messages, setMessages] = useState([]);
    const [inputMessage, setInputMessage] = useState('');
    const [selectedLanguage, setSelectedLanguage] = useState('en');
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
    const [shouldAutoScroll, setShouldAutoScroll] = useState(true);
    const navigate = useNavigate();
    const messagesEndRef = useRef(null);
    const sessionId = useRef(Math.random().toString(36).substring(7));
    const messagesContainerRef = useRef(null);

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
                content: selectedLanguage === 'en' 
                    ? 'Hello! How can I assist you today? Please describe your symptoms.'
                    : 'नमस्ते! मैं आपकी कैसे मदद कर सकता हूं? कृपया अपने लक्षणों का वर्णन करें।',
                timestamp: new Date().toISOString()
            }]);
        }

        // Check microphone permission
        checkMicrophonePermission();
    }, [isStarted, selectedLanguage]);

    useEffect(() => {
        // Load and handle speech synthesis voices
        const loadVoices = () => {
            // Get the list of voices
            const voices = window.speechSynthesis.getVoices();
            if (voices.length > 0) {
                console.log('Voices loaded:', voices.length);
            }
        };

        // Chrome loads voices asynchronously
        if ('speechSynthesis' in window) {
            window.speechSynthesis.onvoiceschanged = loadVoices;
            loadVoices(); // Initial load attempt
        }
    }, []);

    const scrollToBottom = () => {
        if (shouldAutoScroll && messagesEndRef.current) {
            messagesEndRef.current.scrollIntoView({ behavior: "smooth" });
        }
    };

    const handleScroll = () => {
        if (!messagesContainerRef.current) return;
        
        const { scrollTop, scrollHeight, clientHeight } = messagesContainerRef.current;
        const isScrolledToBottom = Math.abs(scrollHeight - clientHeight - scrollTop) < 50;
        setShouldAutoScroll(isScrolledToBottom);
    };

    useEffect(() => {
        const messagesContainer = messagesContainerRef.current;
        if (messagesContainer) {
            messagesContainer.addEventListener('scroll', handleScroll);
        }

        return () => {
            if (messagesContainer) {
                messagesContainer.removeEventListener('scroll', handleScroll);
            }
        };
    }, []);

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    // Text to Speech setup
    const speak = (text, language) => {
        if ('speechSynthesis' in window) {
            // Cancel any ongoing speech
            window.speechSynthesis.cancel();

            const utterance = new SpeechSynthesisUtterance(text);
            
            // Wait for voices to load and configure speech based on language
            const voices = window.speechSynthesis.getVoices();
            if (language === 'hi') {
                // Try to find a Hindi voice
                const hindiVoice = voices.find(voice => 
                    voice.lang.includes('hi') || 
                    voice.name.toLowerCase().includes('hindi')
                );
                if (hindiVoice) {
                    console.log('Using Hindi voice:', hindiVoice.name);
                    utterance.voice = hindiVoice;
                    utterance.lang = 'hi-IN';
                } else {
                    console.log('No Hindi voice found, using default voice with adjusted properties');
                    utterance.rate = 0.9; // Slower for Hindi
                    utterance.pitch = 1;
                    utterance.lang = 'hi-IN';
                }
            } else {
                // Try to find an English voice
                const englishVoice = voices.find(voice => 
                    voice.lang.includes('en-US') || 
                    voice.lang.includes('en-GB')
                );
                if (englishVoice) {
                    console.log('Using English voice:', englishVoice.name);
                    utterance.voice = englishVoice;
                }
                utterance.rate = 1; // Normal speed for English
                utterance.pitch = 1;
                utterance.lang = 'en-US';
            }
            
            utterance.volume = 1;
            
            // Add error handling
            utterance.onerror = (event) => {
                console.error('Speech synthesis error:', event);
            };

            // Speak the text
            window.speechSynthesis.speak(utterance);
        }
    };

    // Typing animation function with simultaneous speech
    const typeMessage = async (message, shouldSpeak = true) => {
        setIsTyping(true);
        let currentText = '';
        const delay = 30; // Delay between each character

        // Start speaking the entire message immediately
        if (shouldSpeak) {
            speak(message, selectedLanguage);
        }

        // Add temporary typing message
        const newMessage = {
            type: 'bot',
            content: '',
            isTyping: true,
            timestamp: new Date().toISOString()
        };
        
        setMessages(prev => [...prev, newMessage]);

        // Type out the message character by character
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

        // Update the final message and ensure it's stored in state
        const finalMessage = {
            type: 'bot',
            content: message,
            timestamp: new Date().toISOString()
        };
        
        return new Promise((resolve) => {
            setMessages(prev => {
                const newMessages = prev.map((msg, index) => {
                    if (index === prev.length - 1 && msg.isTyping) {
                        return finalMessage;
                    }
                    return msg;
                });
                // Save chat history after updating messages
                saveChatHistory();
                resolve();
                return newMessages;
            });
            
            setIsTyping(false);
        });
    };

    const saveChatHistory = async () => {
        try {
            const token = localStorage.getItem('token');
            if (!token) return;

            // Wait for state updates to be reflected
            await new Promise(resolve => setTimeout(resolve, 500));

            // Get the latest messages directly from state
            const currentMessages = messages.map(msg => ({
                ...msg,
                timestamp: msg.timestamp || new Date().toISOString()
            }));

            // Only save if there are messages
            if (currentMessages.length > 0) {
                await axios.post('https://mediassist-4t66.onrender.com/api/chat-history/save', {
                    sessionId: sessionId.current,
                    messages: currentMessages
                }, {
                    headers: { Authorization: `Bearer ${token}` }
                });
                console.log('Chat history saved successfully');
            }
        } catch (error) {
            console.error('Error saving chat history:', error);
        }
    };

    const sendMessageToBackend = async (message, type = 'text', audioBlob = null) => {
        try {
            setIsWaitingForResponse(true);
            let data;
            let headers = {};
            
            // Add user message first for text input
            if (type === 'text') {
                const userMessage = {
                    type: 'user',
                    content: message,
                    timestamp: new Date().toISOString()
                };
                setMessages(prev => [...prev, userMessage]);
                await saveChatHistory(); // Save after user message
            }
            
            if (type === 'voice' && audioBlob) {
                data = new FormData();
                data.append('type', type);
                data.append('session_id', sessionId.current);
                data.append('audio', audioBlob, 'recording.wav');
                data.append('language', selectedLanguage);
                headers['Content-Type'] = 'multipart/form-data';
            } else {
                data = {
                    message: message,
                    type: type,
                    session_id: sessionId.current,
                    language: selectedLanguage
                };
                headers['Content-Type'] = 'application/json';
            }

            const response = await fetch(`${process.env.REACT_APP_API_URL}/api/chat`, {
                method: 'POST',
                headers: headers,
                body: JSON.stringify(data),
            });

            const responseData = await response.json(); // Parse the response

            if (responseData.success) {
                // Only add user message for voice input
                if (type === 'voice' && responseData.recognized_text) {
                    const voiceMessage = {
                        type: 'user',
                        content: responseData.recognized_text,
                        timestamp: new Date().toISOString()
                    };
                    setMessages(prev => [...prev, voiceMessage]);
                    await saveChatHistory(); // Save after voice message
                }

                // Add bot's response with typing animation
                await typeMessage(responseData.message);
                
                // If this is the final message (user said 'no'), ensure everything is saved
                if (message.toLowerCase() === 'no') {
                    // Wait for state updates
                    await new Promise(resolve => setTimeout(resolve, 1000));
                    
                    // If there's a final output message, add it
                    if (responseData.final_output) {
                        const finalMessage = {
                            type: 'bot',
                            content: responseData.final_output,
                            timestamp: new Date().toISOString()
                        };
                        
                        // Update messages with final output
                        setMessages(prev => [...prev, finalMessage]);
                        
                        // Wait for state update and save
                        await new Promise(resolve => setTimeout(resolve, 1000));
                        await saveChatHistory();
                        
                        // Add final diagnosis message
                        const diagnosisMessage = {
                            type: 'system',
                            content: 'Consultation completed. Final assessment has been saved.',
                            timestamp: new Date().toISOString()
                        };
                        setMessages(prev => [...prev, diagnosisMessage]);
                        
                        // Final save
                        await new Promise(resolve => setTimeout(resolve, 1000));
                        await saveChatHistory();
                    }
                }

                if (responseData.is_final) {
                    await saveChatHistory(); // Save before changing session
                    sessionId.current = Math.random().toString(36).substring(7);
                }
            } else {
                throw new Error(responseData.message || 'Error processing request');
            }
        } catch (error) {
            console.error('Error:', error);
            const errorMessage = {
                type: 'system',
                content: selectedLanguage === 'en' 
                    ? 'Error processing your message. Please try again.'
                    : 'आपका संदेश प्रोसेस करने में त्रुटि। कृपया पुनः प्रयास करें।',
                timestamp: new Date().toISOString()
            };
            setMessages(prev => [...prev, errorMessage]);
            await saveChatHistory();
        } finally {
            setIsWaitingForResponse(false);
            await saveChatHistory();
        }
    };

    const handleSendMessage = async (e) => {
        e.preventDefault();
        if (inputMessage.trim() && !isWaitingForResponse) {
            const message = inputMessage.trim();
            setInputMessage(''); // Clear input immediately
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
                // Configure recognition with language support
                recognition.continuous = false;
                recognition.lang = selectedLanguage === 'en' ? 'en-US' : 'hi-IN';  // Set Hindi language for voice input
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
                    let errorMessage = selectedLanguage === 'en' 
                        ? 'Error with voice recognition. ' 
                        : 'आवाज पहचानने में त्रुटि। ';
                    
                    switch(event.error) {
                        case 'network':
                            errorMessage += selectedLanguage === 'en'
                                ? 'Please check your internet connection.'
                                : 'कृपया अपना इंटरनेट कनेक्शन जांचें।';
                            break;
                        case 'not-allowed':
                            errorMessage += selectedLanguage === 'en'
                                ? 'Please allow microphone access in your browser settings.'
                                : 'कृपया अपने ब्राउज़र सेटिंग्स में माइक्रोफ़ोन एक्सेस की अनुमति दें।';
                            break;
                        case 'no-speech':
                            errorMessage += selectedLanguage === 'en'
                                ? 'No speech was detected. Please try again.'
                                : 'कोई आवाज नहीं मिली। कृपया पुनः प्रयास करें।';
                            break;
                        default:
                            errorMessage += selectedLanguage === 'en'
                                ? 'Please try again.'
                                : 'कृपया पुनः प्रयास करें।';
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
                    content: selectedLanguage === 'en' 
                        ? 'Listening... Speak now.'
                        : 'सुन रहा हूं... अब बोलिए।',
                    timestamp: new Date().toISOString()
                }]);

            } catch (error) {
                console.error('Error starting recording:', error);
                setMessages(prev => [...prev, {
                    type: 'system',
                    content: selectedLanguage === 'en'
                        ? 'Error accessing microphone. Please check browser permissions and try again.'
                        : 'माइक्रोफ़ोन तक पहुंचने में त्रुटि। कृपया ब्राउज़र अनुमतियां जांचें और पुनः प्रयास करें।',
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

    const toggleLanguage = () => {
        setSelectedLanguage(prev => prev === 'en' ? 'hi' : 'en');
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

                <div className="menu-section language">
                    <h3 className="menu-title">{!isSidebarCollapsed && 'Language'}</h3>
                    <button onClick={toggleLanguage} className="menu-item">
                        <FaLanguage /> {!isSidebarCollapsed && (selectedLanguage === 'en' ? 'Switch to Hindi' : 'अंग्रेजी में बदलें')}
                    </button>
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

                <div 
                    className="messages-container" 
                    ref={messagesContainerRef}
                    onScroll={handleScroll}
                >
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
                            <div ref={messagesEndRef} />
                        </>
                    )}
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