#!/usr/bin/env python3
"""
Debug script to test login validation
"""

import asyncio
import json

from sqlalchemy import select

from app.core.database import db_manager
from app.core.security import security_manager
from app.models.user import User
from app.schemas.auth import LoginRequest
from app.utils.validation import InputSanitizer


async def debug_login():
    """Debug login process step by step"""
    
    email = "admin@example.com"
    password = "admin123"
    
    print("🔍 Debugging login process...")
    print(f"Email: '{email}'")
    print(f"Password: '{password}'")
    print("-" * 50)
    
    # Step 1: Test email validation (basic check)
    print("1. Testing email format...")
    is_valid_email = "@" in email and "." in email
    print(f"   Email format valid: {is_valid_email}")
    
    # Step 2: Test LoginRequest schema validation
    print("\n2. Testing LoginRequest schema...")
    try:
        login_request = LoginRequest(email=email, password=password)
        print(f"   LoginRequest valid: ✅")
        print(f"   Validated data: {login_request.model_dump()}")
    except Exception as e:
        print(f"   LoginRequest validation failed: ❌")
        print(f"   Error: {e}")
        return
    
    # Step 3: Check if user exists in database
    print("\n3. Checking user in database...")
    try:
        db_manager.initialize()
        
        async with db_manager.get_async_session() as session:
            result = await session.execute(
                select(User).where(User.email == email)
            )
            user = result.scalar_one_or_none()
            
            if user:
                print(f"   User found: ✅")
                print(f"   User ID: {user.id}")
                print(f"   Username: {user.username}")
                print(f"   Email: {user.email}")
                print(f"   Is active: {user.is_active}")
                print(f"   Is verified: {user.is_verified}")
                
                # Step 4: Test password verification
                print("\n4. Testing password verification...")
                is_password_valid = security_manager.verify_password(password, user.hashed_password)
                print(f"   Password valid: {is_password_valid}")
                
                if is_password_valid:
                    print("\n✅ All checks passed! Login should work.")
                else:
                    print("\n❌ Password verification failed!")
                    
            else:
                print(f"   User not found: ❌")
                print("   Run: python quick_test_user.py to create the user")
                
    except Exception as e:
        print(f"   Database error: ❌")
        print(f"   Error: {e}")


if __name__ == "__main__":
    asyncio.run(debug_login())