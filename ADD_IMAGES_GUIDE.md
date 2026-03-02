# How to Add Images to Help Section

## 📁 Image Storage Locations

Images should be placed in the following directories:

1. **Numbers (0-9)**: `static/help_images/numbers/`
2. **Alphabets (A-Z)**: `static/help_images/alphabets/`
3. **Simple Words**: `static/help_images/words/`

## 📝 Naming Convention

**Important:** Name your images using this format: `LABEL_filename.ext`

Examples:
- For number "1": `1_one.jpg` or `1_number1.png`
- For alphabet "A": `A_letterA.jpg` or `A_alphabet.png`
- For word "Hello": `Hello_greeting.jpg` or `Hello_word.png`

The part before the underscore (`_`) will be displayed as the label in the Help section.

## ✅ Method 1: Manual Upload (Easiest)

1. Copy your image files
2. Paste them directly into the appropriate folder:
   - `static/help_images/numbers/` for numbers
   - `static/help_images/alphabets/` for alphabets  
   - `static/help_images/words/` for words
3. Name them using the format: `LABEL_filename.ext`
4. Refresh the Help page in your browser

## ✅ Method 2: Using API (Programmatic)

You can use the API endpoint to upload images. The admin password is set in `app.py` (line 23).

### Using Python:
```python
import requests

url = 'http://localhost:5000/api/upload'
files = {'image': open('path/to/image.jpg', 'rb')}
data = {
    'category': 'numbers',  # or 'alphabets' or 'words'
    'label': '1',
    'password': 'admin123'  # Change this to match your password in app.py
}

response = requests.post(url, files=files, data=data)
print(response.json())
```

### Using curl (Command Line):
```bash
curl -X POST http://localhost:5000/api/upload \
  -F "category=numbers" \
  -F "label=1" \
  -F "image=@path/to/image.jpg" \
  -F "password=admin123"
```

## 📋 Supported Image Formats

- PNG (.png)
- JPEG (.jpg, .jpeg)
- GIF (.gif)
- WebP (.webp)

## 🔍 Example File Structure

```
static/
└── help_images/
    ├── numbers/
    │   ├── 0_zero.jpg
    │   ├── 1_one.jpg
    │   ├── 2_two.png
    │   └── ...
    ├── alphabets/
    │   ├── A_letterA.jpg
    │   ├── B_letterB.jpg
    │   └── ...
    └── words/
        ├── Hello_greeting.jpg
        ├── ThankYou_gratitude.png
        └── ...
```

## ⚠️ Important Notes

1. **Label Extraction**: The label (displayed text) is extracted from the filename. If filename is `A_letterA.jpg`, the label will be `A`.
2. **No Spaces in Label**: Use underscores or no spaces in the label part (before the underscore).
3. **Case Sensitive**: Labels are case-sensitive, so `A` and `a` will be treated as different.
4. **Refresh Required**: After adding images, refresh the Help page to see them.

## 🎯 Quick Start

1. Open the folder: `static/help_images/numbers/` (or alphabets/words)
2. Copy your image file there
3. Rename it to: `LABEL_yourfilename.jpg` (replace LABEL with the actual label like "1", "A", "Hello")
4. Refresh the browser Help page




