import mongoose from 'mongoose';
import dotenv from 'dotenv';

dotenv.config();

const connectDB = async () => {
    try {
        // Clear any existing connections
        await mongoose.disconnect();

        const mongoURI = 'mongodb+srv://root:Pawanmayekar%4010@healthcare.fpmgm.mongodb.net/?retryWrites=true&w=majority&appName=healthcare';
        console.log('Attempting to connect to MongoDB at:', mongoURI);

        const conn = await mongoose.connect(mongoURI, {
            serverSelectionTimeoutMS: 5000,
            socketTimeoutMS: 45000,
        });

        mongoose.connection.on('error', err => {
            console.error('MongoDB connection error:', err);
        });

        mongoose.connection.on('disconnected', () => {
            console.log('MongoDB disconnected');
        });

        mongoose.connection.on('connected', () => {
            console.log('MongoDB Connected successfully');
        });

        return conn;
    } catch (error) {
        console.error('Failed to connect to MongoDB:', error.message);
        throw error; // Let the server handle the error
    }
};

export default connectDB; 