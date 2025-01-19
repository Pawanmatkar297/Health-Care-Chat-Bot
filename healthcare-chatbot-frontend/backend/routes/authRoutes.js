import express from 'express';
import User from '../models/userModel.js';
import jwt from 'jsonwebtoken';
import dotenv from 'dotenv';

dotenv.config();

const router = express.Router();

// Login user
router.post('/login', async (req, res) => {
    try {
        console.log('Login attempt for username:', req.body.username);
        const { username, password } = req.body;

        // Find user
        const user = await User.findOne({ username });
        
        if (user && (await user.matchPassword(password))) {
            // Create token using environment variable
            const token = jwt.sign(
                { userId: user._id },
                process.env.JWT_SECRET || 'your_jwt_secret',
                { expiresIn: '30d' }
            );

            res.json({
                success: true,
                token,
                username: user.username,
                email: user.email
            });
        } else {
            res.status(401).json({
                success: false,
                message: 'Invalid username or password'
            });
        }
    } catch (error) {
        console.error('Login error:', error);
        res.status(500).json({
            success: false,
            message: 'Server error'
        });
    }
});

// Register user
router.post('/signup', async (req, res) => {
    try {
        console.log('Signup request received:', req.body);
        const { username, email, password } = req.body;

        // Check if user exists
        const userExists = await User.findOne({ 
            $or: [
                { email }, 
                { username }
            ] 
        });

        if (userExists) {
            return res.status(400).json({ 
                success: false, 
                message: userExists.email === email ? 'Email already registered' : 'Username already taken'
            });
        }

        // Create new user
        const user = await User.create({
            username,
            email,
            password
        });

        if (user) {
            // Generate token for immediate login
            const token = jwt.sign(
                { userId: user._id },
                process.env.JWT_SECRET || 'your_jwt_secret',
                { expiresIn: '30d' }
            );

            res.status(201).json({
                success: true,
                message: 'User registered successfully',
                data: {
                    username: user.username,
                    email: user.email,
                    token
                }
            });
        }
    } catch (error) {
        console.error('Signup error:', error);
        res.status(500).json({ 
            success: false, 
            message: error.message || 'Server error during signup'
        });
    }
});

export default router;  // Make sure this is the default export