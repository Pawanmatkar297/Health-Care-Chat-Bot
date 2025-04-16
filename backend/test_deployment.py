#!/usr/bin/env python3
import requests
import sys
import time
import json

def test_deployment(url):
    """Test if the deployed chatbot API is working properly"""
    print(f"Testing deployment at {url}")
    
    # Test health check endpoint
    try:
        print("\nTesting health check endpoint...")
        response = requests.get(f"{url}")
        print(f"Status code: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            print("✅ Health check passed")
        else:
            print("❌ Health check failed")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    
    # Test ping endpoint
    try:
        print("\nTesting ping endpoint...")
        response = requests.get(f"{url}/ping")
        print(f"Status code: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200 and response.text == "pong":
            print("✅ Ping test passed")
        else:
            print("❌ Ping test failed")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    
    # Test chat endpoint
    try:
        print("\nTesting chat endpoint...")
        data = {
            "message": "fever and headache",
            "session_id": "test_session",
            "language": "en"
        }
        headers = {
            "Content-Type": "application/json"
        }
        response = requests.post(f"{url}/api/chat", json=data, headers=headers)
        print(f"Status code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        if response.status_code == 200 and response.json().get("success") == True:
            print("✅ Chat test passed")
        else:
            print("❌ Chat test failed")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    url = "https://health-care-chat-bot.onrender.com"
    if len(sys.argv) > 1:
        url = sys.argv[1]
        
    test_deployment(url) 