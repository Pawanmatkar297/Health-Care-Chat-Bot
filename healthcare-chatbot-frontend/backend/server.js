import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import connectDB from './config/db.js';
import authRoutes from './routes/authRoutes.js';

dotenv.config();

const app = express();

// Connect to MongoDB
connectDB();

// Middleware
app.use(cors({
    origin: 'http://localhost:3000',
    credentials: true
}));
app.use(express.json());

// Routes
app.use('/api/auth', authRoutes);

app.get('/api/test', (req, res) => {
    res.json({ message: 'Backend is connected!' });
});

const PORT = process.env.PORT || 5001;

app.listen(PORT, () => {
    console.log(`Auth server running on port ${PORT}`);
}); 