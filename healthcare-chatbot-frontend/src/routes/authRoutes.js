import express from 'express';
import User from '../models/userModel.js';
import jwt from 'jsonwebtoken';

const router = express.Router();

// Register user
router.post('/signup', async (req, res) => {
    try {
        const { username, email, password } = req.body;

        // Check if user exists
        const userExists = await User.findOne({ $or: [{ email }, { username }] });
        if (userExists) {
            return res.status(400).json({ 
                success: false, 
                message: 'User already exists' 
            });
        }

        // Create user
        const user = await User.create({
            username,
            email,
            password
        });

        if (user) {
            res.status(201).json({
                success: true,
                message: 'User registered successfully'
            });
        }
    } catch (error) {
        res.status(500).json({ 
            success: false, 
            message: 'Server error' 
        });
    }
});

// Login user
router.post('/login', async (req, res) => {
    try {
        const { username, password } = req.body;

        // Find user
        const user = await User.findOne({ username });
        
        if (user && (await user.matchPassword(password))) {
            // Create token
            const token = jwt.sign(
                { userId: user._id },
                'your_jwt_secret',  // Replace with your secret key
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
        res.status(500).json({
            success: false,
            message: 'Server error'
        });
    }
});

export default router; 