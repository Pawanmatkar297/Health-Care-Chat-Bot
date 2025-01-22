import express from 'express';
import ChatHistory from '../models/chatHistoryModel.js';
import jwt from 'jsonwebtoken';

const router = express.Router();

// Middleware to verify JWT token
const verifyToken = (req, res, next) => {
    const token = req.headers.authorization?.split(' ')[1];
    
    if (!token) {
        return res.status(401).json({ success: false, message: 'No token provided' });
    }

    try {
        const decoded = jwt.verify(token, process.env.JWT_SECRET || 'your_jwt_secret');
        req.userId = decoded.userId;
        next();
    } catch (error) {
        return res.status(401).json({ success: false, message: 'Invalid token' });
    }
};

// Save chat session
router.post('/save', verifyToken, async (req, res) => {
    try {
        const { sessionId, messages } = req.body;
        const userId = req.userId;

        const chatHistory = new ChatHistory({
            userId,
            sessionId,
            messages
        });

        await chatHistory.save();
        res.status(201).json({ success: true, message: 'Chat history saved successfully' });
    } catch (error) {
        console.error('Error saving chat history:', error);
        res.status(500).json({ success: false, message: 'Error saving chat history' });
    }
});

// Get all chat sessions for a user
router.get('/sessions', verifyToken, async (req, res) => {
    try {
        const userId = req.userId;
        const chatHistories = await ChatHistory.find({ userId })
            .sort({ createdAt: -1 })
            .select('sessionId createdAt messages');
            
        res.json({ success: true, chatHistories });
    } catch (error) {
        console.error('Error fetching chat histories:', error);
        res.status(500).json({ success: false, message: 'Error fetching chat histories' });
    }
});

// Get specific chat session
router.get('/session/:sessionId', verifyToken, async (req, res) => {
    try {
        const userId = req.userId;
        const { sessionId } = req.params;
        
        const chatHistory = await ChatHistory.findOne({ userId, sessionId });
        if (!chatHistory) {
            return res.status(404).json({ success: false, message: 'Chat session not found' });
        }
        
        res.json({ success: true, chatHistory });
    } catch (error) {
        console.error('Error fetching chat session:', error);
        res.status(500).json({ success: false, message: 'Error fetching chat session' });
    }
});

export default router; 