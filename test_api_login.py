#!/usr/bin/env python3
"""
Test the login API endpoint directly
"""

import asyncio
import json

import httpx


async def test_login_api():
    """Test the login API endpoint directly"""
    
    url = "http://localhost:8000/api/v1/auth/login"
    data = {
        "email": "admin@example.com",
        "password": "admin123"
    }
    
    print("🧪 Testing login API endpoint directly...")
    print(f"URL: {url}")
    print(f"Data: {data}")
    print("-" * 50)
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                url,
                json=data,
                headers={"Content-Type": "application/json"}
            )
            
            print(f"Status Code: {response.status_code}")
            print(f"Headers: {dict(response.headers)}")
            print("-" * 50)
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Login successful!")
                print(f"Response: {json.dumps(result, indent=2)}")
            else:
                print("❌ Login failed!")
                try:
                    error_data = response.json()
                    print(f"Error response: {json.dumps(error_data, indent=2)}")
                except:
                    print(f"Raw response: {response.text}")
                    
        except Exception as e:
            print(f"❌ Request failed: {e}")


if __name__ == "__main__":
    asyncio.run(test_login_api())