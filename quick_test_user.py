#!/usr/bin/env python3
"""
Quick script to create a test user for login testing
"""

import asyncio
from datetime import datetime

from sqlalchemy import select

from app.core.database import db_manager
from app.core.security import security_manager
from app.models.user import User


async def create_quick_test_user():
    """Create a test user quickly"""
    
    # Test credentials
    username = "admin"
    email = "admin@example.com"
    password = "admin123"
    
    print("Creating test user...")
    print(f"Username: {username}")
    print(f"Password: {password}")
    
    # Initialize database
    db_manager.initialize()
    
    async with db_manager.get_async_session() as session:
        # Check if user exists
        result = await session.execute(
            select(User).where(User.username == username)
        )
        existing_user = result.scalar_one_or_none()
        
        if existing_user:
            print("✅ Test user already exists!")
            return
        
        # Create user
        hashed_password = security_manager.hash_password(password)
        
        new_user = User(
            username=username,
            email=email,
            hashed_password=hashed_password,
            full_name="Admin User",
            is_active=True,
            is_verified=True,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        session.add(new_user)
        await session.commit()
        
        print("✅ Test user created!")
        print("You can now login with:")
        print(f"  Username: {username}")
        print(f"  Password: {password}")


if __name__ == "__main__":
    asyncio.run(create_quick_test_user())