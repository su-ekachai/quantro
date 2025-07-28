#!/usr/bin/env python3
"""
Script to create a test user for the Quantro trading platform
"""

import asyncio
from datetime import datetime

from sqlalchemy import select

from app.core.database import db_manager
from app.core.security import security_manager
from app.models.user import User


async def create_test_user():
    """Create a test user for development and testing"""
    
    # Test user credentials
    username = "testuser"
    email = "test@example.com"
    password = "password123"
    full_name = "Test User"
    
    print("🚀 Creating test user for Quantro Trading Platform...")
    print(f"Username: {username}")
    print(f"Email: {email}")
    print(f"Password: {password}")
    print("-" * 50)
    
    try:
        # Initialize database and get session
        db_manager.initialize()
        
        async with db_manager.get_async_session() as session:
            # Check if user already exists
            result = await session.execute(
                select(User).where(User.username == username)
            )
            existing_user = result.scalar_one_or_none()
            
            if existing_user:
                print(f"❌ User '{username}' already exists!")
                print("To reset the password, delete the user first or use a different username.")
                return False
            
            # Hash the password
            hashed_password = security_manager.hash_password(password)
            
            # Create new user
            new_user = User(
                username=username,
                email=email,
                hashed_password=hashed_password,
                full_name=full_name,
                preferred_language="en",
                theme="light",
                timezone="UTC",
                is_active=True,
                is_verified=True,
                failed_login_attempts=0,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            
            # Add to database
            session.add(new_user)
            await session.commit()
            await session.refresh(new_user)
            
            print(f"✅ Test user created successfully!")
            print(f"User ID: {new_user.id}")
            print(f"Created at: {new_user.created_at}")
            print("-" * 50)
            print("🧪 You can now test login with:")
            print(f"   Username: {username}")
            print(f"   Password: {password}")
            print("-" * 50)
            
            return True
            
    except Exception as e:
        print(f"❌ Error creating test user: {e}")
        return False


async def delete_test_user():
    """Delete the test user (useful for cleanup)"""
    username = "testuser"
    
    print(f"🗑️  Deleting test user '{username}'...")
    
    try:
        db_manager.initialize()
        
        async with db_manager.get_async_session() as session:
            # Find the user
            result = await session.execute(
                select(User).where(User.username == username)
            )
            user = result.scalar_one_or_none()
            
            if not user:
                print(f"❌ User '{username}' not found!")
                return False
            
            # Delete the user
            await session.delete(user)
            await session.commit()
            
            print(f"✅ Test user '{username}' deleted successfully!")
            return True
            
    except Exception as e:
        print(f"❌ Error deleting test user: {e}")
        return False


async def list_users():
    """List all users in the database"""
    print("👥 Listing all users...")
    
    try:
        db_manager.initialize()
        
        async with db_manager.get_async_session() as session:
            result = await session.execute(select(User))
            users = result.scalars().all()
            
            if not users:
                print("No users found in the database.")
                return
            
            print(f"Found {len(users)} user(s):")
            print("-" * 80)
            print(f"{'ID':<5} {'Username':<15} {'Email':<25} {'Active':<8} {'Created'}")
            print("-" * 80)
            
            for user in users:
                print(f"{user.id:<5} {user.username:<15} {user.email:<25} {user.is_active:<8} {user.created_at.strftime('%Y-%m-%d %H:%M')}")
            
    except Exception as e:
        print(f"❌ Error listing users: {e}")


async def main():
    """Main function with menu options"""
    print("🔐 Quantro Test User Management")
    print("=" * 40)
    print("1. Create test user")
    print("2. Delete test user")
    print("3. List all users")
    print("4. Exit")
    print("=" * 40)
    
    while True:
        choice = input("Enter your choice (1-4): ").strip()
        
        if choice == "1":
            await create_test_user()
            break
        elif choice == "2":
            await delete_test_user()
            break
        elif choice == "3":
            await list_users()
            break
        elif choice == "4":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")


if __name__ == "__main__":
    asyncio.run(main())