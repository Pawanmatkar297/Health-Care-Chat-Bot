import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import axios from 'axios';
import './Login.css';
import { FaUser, FaLock, FaUserMd, FaArrowRight, FaRobot } from 'react-icons/fa';

const Login = () => {
    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');
    const [error, setError] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const navigate = useNavigate();

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError('');
        setIsLoading(true);

        try {
            const response = await axios.post('https://mediassist-4t66.onrender.com/api/auth/login',
                {
                    username,
                    password
                },
                {
                    headers: {
                        'Content-Type': 'application/json',
                        'Accept': 'application/json'
                    }
                }
            );

            if (response.data.success) {
                localStorage.setItem('token', response.data.token);
                localStorage.setItem('username', username);
                navigate('/chat');
            } else {
                setError(response.data.message || 'Login failed');
            }
        } catch (error) {
            setError(error.response?.data?.message || 'An error occurred during login');
        } finally {
            setIsLoading(false);
        }
    };

    const handleGuestAccess = () => {
        localStorage.setItem('token', 'guest-token');
        localStorage.setItem('username', 'Guest');
        navigate('/chat');
    };

    return (
        <div className="login-page">
            <div className="login-container">
                <div className="login-card">
                    <div className="login-header">
                        <div className="logo-container">
                        <img src="/bot_assistant_QWK_icon.ico" alt="MedAssist Logo" className="logo" />
                            <h1>MedAssist</h1>
                        </div>
                        <p>Sign in to access your healthcare chatbot</p>
                    </div>

                    <form onSubmit={handleSubmit}>
                        <div className="form-group">
                            <label htmlFor="username">
                                <FaUser /> Username
                            </label>
                            <input
                                type="text"
                                id="username"
                                value={username}
                                onChange={(e) => setUsername(e.target.value)}
                                placeholder="Enter your username"
                                required
                            />
                        </div>

                        <div className="form-group">
                            <label htmlFor="password">
                                <FaLock /> Password
                            </label>
                            <input
                                type="password"
                                id="password"
                                value={password}
                                onChange={(e) => setPassword(e.target.value)}
                                placeholder="Enter your password"
                                required
                            />
                        </div>

                        {error && <div className="error-message">{error}</div>}

                        <button 
                            type="submit" 
                            className="login-button" 
                            disabled={isLoading}
                        >
                            {isLoading ? (
                                <span className="loading-spinner"></span>
                            ) : (
                                'Sign In'
                            )}
                        </button>
                    </form>

                    <button 
                        onClick={handleGuestAccess}
                        className="guest-button"
                        type="button"
                    >
                        Continue as Guest <FaArrowRight />
                    </button>

                    <div className="register-link">
                        Don't have an account? <Link to="/register">Register here</Link>
                    </div>
                </div>
            </div>

            <div className="features-grid">
                <div className="feature">
                    <FaUserMd className="feature-icon" />
                    <h3>24/7 Healthcare Support</h3>
                    <p>Access medical assistance anytime, anywhere</p>
                </div>
                <div className="feature">
                    <FaRobot className="feature-icon" />
                    <h3>AI-Powered Assistant</h3>
                    <p>Get instant responses to your health queries</p>
                </div>
            </div>
        </div>
    );
};

export default Login; 