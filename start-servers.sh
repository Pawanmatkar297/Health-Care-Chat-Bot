#!/bin/bash

# Start the Node.js server
cd healthcare-chatbot-frontend/backend
npm start &

# Start the Flask server
cd ../..
cd backend
python app.py &

# Start the React frontend
cd ../healthcare-chatbot-frontend
npm start 