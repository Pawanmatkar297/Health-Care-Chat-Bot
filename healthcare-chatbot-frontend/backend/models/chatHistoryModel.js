import mongoose from 'mongoose';

const messageSchema = new mongoose.Schema({
    type: {
        type: String,
        required: true,
        enum: ['user', 'bot', 'system']
    },
    content: {
        type: String,
        required: true
    },
    timestamp: {
        type: Date,
        default: Date.now
    }
});

const chatHistorySchema = new mongoose.Schema({
    userId: {
        type: mongoose.Schema.Types.ObjectId,
        ref: 'User',
        required: true
    },
    sessionId: {
        type: String,
        required: true
    },
    messages: [messageSchema],
    createdAt: {
        type: Date,
        default: Date.now
    }
});

const ChatHistory = mongoose.model('ChatHistory', chatHistorySchema);
export default ChatHistory; 