"""
Simple script to upload images to the Help section
Usage: python upload_image.py
"""

import requests
import os
import sys

# Configuration
BASE_URL = "http://localhost:5000"
ADMIN_PASSWORD = "admin123"  # Change this to match your password in app.py

def upload_image(image_path, category, label):
    """
    Upload an image to the help section
    
    Args:
        image_path: Path to the image file
        category: 'numbers', 'alphabets', or 'words'
        label: The label to display (e.g., '1', 'A', 'Hello')
    """
    if not os.path.exists(image_path):
        print(f"❌ Error: Image file not found: {image_path}")
        return False
    
    if category not in ['numbers', 'alphabets', 'words']:
        print(f"❌ Error: Category must be 'numbers', 'alphabets', or 'words'")
        return False
    
    url = f"{BASE_URL}/api/upload"
    
    try:
        with open(image_path, 'rb') as f:
            files = {'image': f}
            data = {
                'category': category,
                'label': label,
                'password': ADMIN_PASSWORD
            }
            
            response = requests.post(url, files=files, data=data)
            result = response.json()
            
            if result.get('success'):
                print(f"✅ Successfully uploaded: {label} to {category}")
                return True
            else:
                print(f"❌ Error: {result.get('message', 'Unknown error')}")
                return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("ISL Help Section - Image Uploader")
    print("=" * 50)
    print()
    
    # Get input from user
    image_path = input("Enter image file path: ").strip().strip('"')
    print("\nCategories:")
    print("1. numbers")
    print("2. alphabets")
    print("3. words")
    category_choice = input("\nEnter category (1/2/3): ").strip()
    
    category_map = {'1': 'numbers', '2': 'alphabets', '3': 'words'}
    category = category_map.get(category_choice)
    
    if not category:
        print("❌ Invalid category choice")
        sys.exit(1)
    
    label = input(f"Enter label for {category} (e.g., '1', 'A', 'Hello'): ").strip()
    
    if not label:
        print("❌ Label cannot be empty")
        sys.exit(1)
    
    print()
    upload_image(image_path, category, label)




