import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import connectDB from './config/db.js';
import authRoutes from './routes/authRoutes.js';
import chatHistoryRoutes from './routes/chatHistoryRoutes.js';

dotenv.config();

const app = express();

// Connect to MongoDB
connectDB()
    .then(() => console.log('Database connected successfully'))
    .catch(err => {
        console.error('Database connection error:', err);
        process.exit(1);
    });

// CORS configuration
app.use(cors({
    origin: ['http://localhost:3000', 'http://localhost:5000'],
    credentials: false,
    methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
    allowedHeaders: ['Content-Type', 'Authorization', 'Origin', 'Accept'],
}));

// Handle preflight requests
app.options('*', cors());

app.use(express.json());

// Error handling middleware
app.use((err, req, res, next) => {
    console.error(err.stack);
    res.status(500).json({ success: false, message: 'Something broke!' });
});

// Routes
app.use('/api/auth', authRoutes);
app.use('/api/chat-history', chatHistoryRoutes);

// Test route
app.get('/api/test', (req, res) => {
    res.json({ message: 'Backend is connected!' });
});

const PORT = 5001;

app.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`);
    console.log(`Test the connection at: http://localhost:${PORT}/api/test`);
}); 