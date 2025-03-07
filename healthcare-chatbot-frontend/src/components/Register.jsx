import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { FaUser, FaLock, FaEnvelope, FaRobot, FaUserMd } from 'react-icons/fa';
import axios from 'axios';
import './Register.css';

const Register = () => {
    const [formData, setFormData] = useState({
        username: '',
        email: '',
        password: '',
        confirmPassword: ''
    });
    const [error, setError] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const navigate = useNavigate();

    const handleChange = (e) => {
        setFormData({
            ...formData,
            [e.target.name]: e.target.value
        });
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError('');
        setIsLoading(true);

        // Password validation
        if (formData.password !== formData.confirmPassword) {
            setError('Passwords do not match');
            setIsLoading(false);
            return;
        }

        if (formData.password.length < 6) {
            setError('Password must be at least 6 characters long');
            setIsLoading(false);
            return;
        }

        try {
            const response = await axios.post(`${process.env.REACT_APP_AUTH_URL}/api/auth/signup`, 
                {
                    username: formData.username,
                    email: formData.email,
                    password: formData.password
                },
                {
                    headers: {
                        'Content-Type': 'application/json',
                        'Accept': 'application/json'
                    }
                }
            );

            if (response.data.success) {
                // Automatically log in the user
                localStorage.setItem('token', response.data.token);
                localStorage.setItem('username', formData.username);
                navigate('/chat');
            } else {
                setError(response.data.message || 'Registration failed');
            }
        } catch (error) {
            setError(error.response?.data?.message || 'An error occurred during registration');
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="register-page">
            <div className="register-container">
                <div className="register-card">
                    <div className="register-header">
                        <div className="logo-container">
                        <img src="/bot_assistant_QWK_icon.ico" alt="MedAssist Logo" className="rlogo" />
                            <h1>MedAssist</h1>
                        </div>
                        <p>Create your account to get started</p>
                    </div>

                    <form onSubmit={handleSubmit}>
                        <div className="form-group">
                            <label htmlFor="username">
                                <FaUser /> Username
                            </label>
                            <input
                                type="text"
                                id="username"
                                name="username"
                                value={formData.username}
                                onChange={handleChange}
                                placeholder="Choose a username"
                                required
                            />
                        </div>

                        <div className="form-group">
                            <label htmlFor="email">
                                <FaEnvelope /> Email
                            </label>
                            <input
                                type="email"
                                id="email"
                                name="email"
                                value={formData.email}
                                onChange={handleChange}
                                placeholder="Enter your email"
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
                                name="password"
                                value={formData.password}
                                onChange={handleChange}
                                placeholder="Create a password"
                                required
                            />
                        </div>

                        <div className="form-group">
                            <label htmlFor="confirmPassword">
                                <FaLock /> Confirm Password
                            </label>
                            <input
                                type="password"
                                id="confirmPassword"
                                name="confirmPassword"
                                value={formData.confirmPassword}
                                onChange={handleChange}
                                placeholder="Confirm your password"
                                required
                            />
                        </div>

                        {error && <div className="error-message">{error}</div>}

                        <button 
                            type="submit" 
                            className="register-button"
                            disabled={isLoading}
                        >
                            {isLoading ? (
                                <span className="loading-spinner"></span>
                            ) : (
                                'Create Account'
                            )}
                        </button>
                    </form>

                    <div className="login-link">
                        Already have an account? <Link to="/login">Sign in here</Link>
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

export default Register; 