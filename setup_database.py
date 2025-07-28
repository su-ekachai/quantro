#!/usr/bin/env python3
"""
Setup script to initialize database and create test user
"""

import asyncio
import subprocess
import sys
from datetime import datetime

from sqlalchemy import select, text

from app.core.database import db_manager
from app.core.security import security_manager
from app.models.user import User


async def check_database_connection():
    """Check if database connection works"""
    print("🔍 Checking database connection...")
    
    try:
        db_manager.initialize()
        
        async with db_manager.get_async_session() as session:
            result = await session.execute(text("SELECT 1"))
            result.scalar()
            print("✅ Database connection successful!")
            return True
            
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False


def run_migrations():
    """Run Alembic migrations"""
    print("🔄 Running database migrations...")
    
    try:
        result = subprocess.run(
            ["alembic", "upgrade", "head"],
            capture_output=True,
            text=True,
            check=True
        )
        print("✅ Migrations completed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Migration failed: {e}")
        print(f"Error output: {e.stderr}")
        return False
    except FileNotFoundError:
        print("❌ Alembic not found. Make sure it's installed: pip install alembic")
        return False


async def create_admin_user():
    """Create admin user for testing"""
    print("👤 Creating admin user...")
    
    username = "admin"
    email = "admin@example.com"
    password = "admin123"
    
    try:
        async with db_manager.get_async_session() as session:
            # Check if user exists
            result = await session.execute(
                select(User).where(User.username == username)
            )
            existing_user = result.scalar_one_or_none()
            
            if existing_user:
                print(f"✅ Admin user already exists!")
                return True
            
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
            
            print("✅ Admin user created successfully!")
            print(f"   Username: {username}")
            print(f"   Password: {password}")
            return True
            
    except Exception as e:
        print(f"❌ Error creating admin user: {e}")
        return False


async def main():
    """Main setup function"""
    print("🚀 Quantro Database Setup")
    print("=" * 40)
    
    # Step 1: Check database connection
    if not await check_database_connection():
        print("\n💡 Make sure PostgreSQL is running and the database 'quantro' exists.")
        print("   You can create it with: createdb quantro")
        return False
    
    # Step 2: Run migrations
    if not run_migrations():
        print("\n💡 Migration failed. Check your database configuration.")
        return False
    
    # Step 3: Create admin user
    if not await create_admin_user():
        print("\n💡 Failed to create admin user.")
        return False
    
    print("\n🎉 Setup completed successfully!")
    print("=" * 40)
    print("You can now:")
    print("1. Start the server: uvicorn app.main:app --reload")
    print("2. Visit: http://localhost:8000/login")
    print("3. Login with: admin / admin123")
    print("=" * 40)
    
    return True


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)