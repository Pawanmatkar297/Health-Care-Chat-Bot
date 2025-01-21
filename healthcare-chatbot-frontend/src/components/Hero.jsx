import React, { useEffect } from 'react';
import { Link } from 'react-router-dom';
import { FaRobot, FaUserMd, FaHeartbeat, FaClock, FaArrowRight, FaShieldAlt, FaBrain } from 'react-icons/fa';
import './Hero.css';

const Hero = () => {
    useEffect(() => {
        const observerOptions = {
            threshold: 0.1,
            rootMargin: "0px"
        };

        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('animate-in');
                }
            });
        }, observerOptions);

        document.querySelectorAll('.animate-on-scroll').forEach((elem) => observer.observe(elem));

        return () => observer.disconnect();
    }, []);

    return (
        <div className="hero-container">
            {/* Animated Background */}
            <div className="animated-bg">
                <div className="gradient-sphere"></div>
                <div className="gradient-sphere secondary"></div>
            </div>

            {/* Main Hero Section */}
            <div className="hero-main">
                <div className="hero-content animate-on-scroll">
                    <div className="hero-logo">
                        <div className="logo-wrapper">
                            <img src="/bot_assistant_QWK_icon.ico" alt="MedAssist Logo" className="logo" />
                            <div className="logo-glow"></div>
                        </div>
                        <h1><span className="gradient-text">MedAssist</span></h1>
                    </div>
                    <p className="hero-tagline">Your AI-Powered Healthcare Companion</p>
                    <p className="hero-description">
                        Experience next-generation healthcare support powered by advanced AI. 
                        Get instant medical guidance, real-time symptom analysis, and personalized health recommendations 24/7.
                    </p>
                    <div className="hero-cta">
                        <Link to="/register" className="cta-button primary">
                            Get Started
                            <span className="button-glow"></span>
                        </Link>
                        <Link to="/login" className="cta-button secondary">Sign In</Link>
                    </div>
                    <div className="trust-indicators">
                        <div className="trust-item">
                            <FaShieldAlt /> HIPAA Compliant
                        </div>
                        <div className="trust-item">
                            <FaBrain /> Advanced AI
                        </div>
                        <div className="trust-item">
                            <FaHeartbeat /> 24/7 Support
                        </div>
                    </div>
                </div>
            </div>

            {/* Features Section */}
            <div className="features-section">
                <h2 className="section-title animate-on-scroll">Why Choose Our Platform</h2>
                <div className="features-grid">
                    <div className="feature-card animate-on-scroll">
                        <div className="feature-icon-wrapper">
                            <FaRobot className="feature-icon" />
                        </div>
                        <h3>AI-Powered Analysis</h3>
                        <p>State-of-the-art algorithms provide accurate symptom analysis and health recommendations</p>
                        <div className="feature-hover-effect"></div>
                    </div>
                    <div className="feature-card animate-on-scroll" style={{"--delay": "0.2s"}}>
                        <div className="feature-icon-wrapper">
                            <FaUserMd className="feature-icon" />
                        </div>
                        <h3>Medical Expertise</h3>
                        <p>Built with comprehensive medical knowledge and up-to-date healthcare information</p>
                        <div className="feature-hover-effect"></div>
                    </div>
                    <div className="feature-card animate-on-scroll" style={{"--delay": "0.4s"}}>
                        <div className="feature-icon-wrapper">
                            <FaClock className="feature-icon" />
                        </div>
                        <h3>24/7 Availability</h3>
                        <p>Access healthcare support anytime, anywhere with instant, reliable responses</p>
                        <div className="feature-hover-effect"></div>
                    </div>
                    <div className="feature-card animate-on-scroll" style={{"--delay": "0.6s"}}>
                        <div className="feature-icon-wrapper">
                            <FaHeartbeat className="feature-icon" />
                        </div>
                        <h3>Personalized Care</h3>
                        <p>Tailored health insights based on your unique symptoms and medical history</p>
                        <div className="feature-hover-effect"></div>
                    </div>
                </div>
            </div>

            {/* How It Works Section */}
            <div className="how-it-works">
                <h2 className="section-title animate-on-scroll">How It Works</h2>
                <div className="steps-container">
                    <div className="step animate-on-scroll">
                        <div className="step-number">1</div>
                        <div className="step-content">
                            <h3>Sign Up</h3>
                            <p>Create your secure account in seconds</p>
                        </div>
                        <div className="step-background"></div>
                    </div>
                    <div className="step-arrow animate-on-scroll"><FaArrowRight /></div>
                    <div className="step animate-on-scroll" style={{"--delay": "0.2s"}}>
                        <div className="step-number">2</div>
                        <div className="step-content">
                            <h3>Describe Symptoms</h3>
                            <p>Share your health concerns</p>
                        </div>
                        <div className="step-background"></div>
                    </div>
                    <div className="step-arrow animate-on-scroll"><FaArrowRight /></div>
                    <div className="step animate-on-scroll" style={{"--delay": "0.4s"}}>
                        <div className="step-number">3</div>
                        <div className="step-content">
                            <h3>Get Analysis</h3>
                            <p>Receive AI-powered insights</p>
                        </div>
                        <div className="step-background"></div>
                    </div>
                </div>
            </div>

            {/* About Platform Section */}
            <div className="about-platform">
                <div className="about-grid">
                    <div className="about-content animate-on-scroll">
                        <h2 className="section-title">About Our Platform</h2>
                        <p>
                            Our Healthcare Assistant represents the convergence of cutting-edge AI technology 
                            and medical expertise. We're revolutionizing healthcare accessibility while maintaining 
                            the highest standards of accuracy, privacy, and user experience.
                        </p>
                        <div className="about-cta">
                            <Link to="/register" className="cta-button primary">
                                Start Your Health Journey
                                <span className="button-glow"></span>
                            </Link>
                        </div>
                    </div>
                    <div className="about-stats animate-on-scroll">
                        <div className="stat-card">
                            <div className="stat-number">99%</div>
                            <div className="stat-label">Accuracy Rate</div>
                        </div>
                        <div className="stat-card">
                            <div className="stat-number">24/7</div>
                            <div className="stat-label">Availability</div>
                        </div>
                        <div className="stat-card">
                            <div className="stat-number">100K+</div>
                            <div className="stat-label">Users Helped</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default Hero; 