# Indian Sign Language Recognition System

A professional real-time Indian Sign Language (ISL) recognition system with a modern web interface. Uses MediaPipe for hand detection and a trained neural network for gesture classification.

## 🚀 Features

- **Real-time Recognition**: Live two-handed gesture recognition
- **Professional Web Interface**: Modern, responsive design
- **Easy Controls**: Start/Stop/Exit prediction with one click
- **Learning Resources**: Integrated ISL tutorial links
- **Mobile Responsive**: Works on all devices

## 📋 Requirements

- Python 3.7+
- OpenCV
- MediaPipe
- TensorFlow
- Flask
- Flask-CORS

## 🛠️ Installation

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application:**
   ```bash
   python app.py
   ```

3. **Open your browser:**
   - Go to `http://localhost:5000`
   - Allow camera access when prompted

## 🎯 Usage

1. **Start Prediction**: Click "Start Prediction" to begin recognition
2. **Stop Prediction**: Click "Stop Prediction" to pause
3. **Exit**: Click "Exit" to close the application
4. **Learn ISL**: Use the learning resource cards for tutorials

## 📁 Project Structure

```
ISL_Interpreter_2Hands/
├── app.py                    # Main Flask application
├── simple_frontend.html      # Professional web interface
├── requirements.txt          # Python dependencies
├── model/                    # Trained ML models
│   ├── isl_model_two.keras
│   └── label_encoder_two.pkl
├── data/                     # Training data
│   └── isl_landmarks_two.csv
├── static/                   # CSS styles
├── templates/                # Flask templates
└── utils_preproc_two.py      # Data preprocessing utilities
```

## 🔧 Development

- **Training**: `python train_model_two_hands.py`
- **Data Collection**: `python collect_data_two_hands.py`
- **Real-time Testing**: `python realtime_isl_two_hands.py`

## 🎨 Interface

The web interface features:
- **Dark Professional Theme**: Modern, business-ready design
- **Real-time Video Feed**: Live camera input with gesture overlay
- **Control Panel**: Intuitive start/stop controls
- **Learning Section**: Direct links to ISL tutorials
- **Responsive Design**: Works on desktop and mobile

## 📱 Browser Support

- Chrome (recommended)
- Firefox
- Safari
- Edge

## 🔒 Privacy

- All processing happens locally
- No data is sent to external servers
- Camera access is only used for gesture recognition