#!/usr/bin/env python3
"""
Fix the existing user's email to be valid
"""

import asyncio

from sqlalchemy import update

from app.core.database import db_manager
from app.models.user import User


async def fix_user_email():
    """Fix the existing admin user's email"""
    
    print("🔧 Fixing admin user email...")
    
    try:
        db_manager.initialize()
        
        async with db_manager.get_async_session() as session:
            # Update the admin user's email
            result = await session.execute(
                update(User)
                .where(User.username == "admin")
                .values(email="admin@example.com")
            )
            
            await session.commit()
            
            if result.rowcount > 0:
                print("✅ Admin user email updated to admin@example.com")
            else:
                print("❌ Admin user not found")
                
    except Exception as e:
        print(f"❌ Error fixing user email: {e}")


if __name__ == "__main__":
    asyncio.run(fix_user_email())