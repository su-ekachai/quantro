#!/usr/bin/env python3
"""
Script to reset database and create fresh migrations
"""

import subprocess
import sys
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            check=True
        )
        print(f"✅ {description} completed!")
        if result.stdout:
            print(f"   Output: {result.stdout.strip()}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed!")
        print(f"   Error: {e.stderr.strip()}")
        return False


def main():
    """Reset database and migrations"""
    print("🔄 Resetting Quantro Database and Migrations")
    print("=" * 50)
    
    # Step 1: Drop and recreate database
    print("1. Resetting database...")
    if not run_command("dropdb quantro --if-exists", "Dropping existing database"):
        print("   (This is OK if database didn't exist)")
    
    if not run_command("createdb quantro", "Creating fresh database"):
        print("❌ Failed to create database. Make sure PostgreSQL is running.")
        return False
    
    # Step 2: Remove alembic version tracking
    print("\n2. Clearing Alembic version tracking...")
    versions_dir = Path("alembic/versions")
    if versions_dir.exists():
        for file in versions_dir.glob("*.py"):
            if file.name != "__init__.py":
                file.unlink()
                print(f"   Removed: {file.name}")
    
    # Clear pycache
    pycache_dir = versions_dir / "__pycache__"
    if pycache_dir.exists():
        for file in pycache_dir.glob("*"):
            file.unlink()
        pycache_dir.rmdir()
        print("   Cleared __pycache__")
    
    # Step 3: Create new initial migration
    print("\n3. Creating fresh initial migration...")
    if not run_command(
        'alembic revision --autogenerate -m "Initial migration"',
        "Generating initial migration"
    ):
        return False
    
    # Step 4: Apply migration
    print("\n4. Applying migration...")
    if not run_command("alembic upgrade head", "Applying migration to database"):
        return False
    
    print("\n🎉 Database reset completed successfully!")
    print("=" * 50)
    print("Next steps:")
    print("1. Run: python setup_database.py")
    print("2. Or run: python quick_test_user.py")
    print("3. Start server: uvicorn app.main:app --reload")
    print("=" * 50)
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)