# 🚀 Nova.AI Desktop Widget
A high-performance desktop application that provides real-time meeting transcription, intelligent summarization, and AI-powered response suggestions with professional-grade reliability and speed.

## ✨ Enhanced Features
### 🎯 Core Capabilities
- **⚡ Real-Time Transcription**: Live speech-to-text with 5-second chunks
- **🎤 Microphone Selection**: Support for multiple input devices
- **🧠 AI Summarization**: Structured summaries with key points and action items  
- **💡 Smart Suggestions**: Context-aware response recommendations with confidence indicators
- **🚀 Performance Optimized**: Async processing, caching, retry logic
- **🔒 Privacy Focused**: Local processing with secure backend connection
- **📊 Performance Monitoring**: Real-time metrics and connection status

### 🛠 Technical Improvements
- **Async Architecture**: Non-blocking operations with thread pools
- **Smart Caching**: Reduces API calls and improves response times
- **Error Recovery**: Automatic retries with exponential backoff
- **Health Monitoring**: Real-time system status and performance metrics
- **Modern UI**: Clean interface with progress indicators
- **Standalone Application**: No browser dependencies

## 🏗 Architecture Overview
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Desktop Widget  │────│  FastAPI Backend │────│   Groq API      │
│                 │    │                  │    │                 │
│ • Audio Capture │    │ • Async Processing│    │ • Whisper       │
│ • UI Management │    │ • Smart Caching  │    │ • Llama Models  │
│ • Error Handling│    │ • Health Checks  │    │ • Rate Limiting │
│ • Device Select │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## ⚡ Quick Start
### 1. Backend Setup
#### Option A: Docker (Recommended)
```bash
# Clone the repository
git clone https://github.com/yourusername/nova-ai-desktop.git
cd nova-ai-desktop

# Build and run the backend
docker build -t nova-ai-backend .
docker run -d -p 8000:8000 --env-file .env nova-ai-backend
```

#### Option B: Direct Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/nova-ai-desktop.git
cd nova-ai-desktop/backend

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export GROQ_API_KEY=gsk_your_key_here
export ENVIRONMENT=production
export PORT=8000

# Start the backend
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### 2. Desktop Widget Setup
#### Option A: Pre-built Executable (Recommended)
1. Download the latest release for your operating system
2. Install the application following the on-screen instructions
3. Launch Nova.AI from your applications menu

#### Option B: From Source
```bash
# Clone the repository
git clone https://github.com/yourusername/nova-ai-desktop.git
cd nova-ai-desktop

# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py
```

## 📋 Usage Guide
### Starting a Transcription Session
1. **Configure Microphone**
   - Launch the Nova.AI Desktop Widget
   - Select your preferred microphone from the dropdown
   - Ensure the backend connection status shows "Connected"

2. **Start Recording**
   - Click "Start Recording" or press Ctrl+R
   - The status will change to "Recording..."
   - Begin speaking or join your meeting

3. **Monitor Progress**
   - Watch live transcription appear in the main window
   - Track chunk count and processing time
   - View connection status in the status bar

### Using AI Features
1. **Generate Summary**
   - After sufficient audio has been captured, a summary will automatically appear
   - Review structured summary with:
     - Key discussion points
     - Action items identified
     - Important decisions made

2. **Get Response Suggestions**
   - AI-powered suggestions appear automatically
   - View context and confidence level
   - Suggestions update as more conversation is captured

3. **Export Transcript**
   - Click "Export" to save your transcript
   - Files are saved with timestamp for easy organization

## ⚙️ Configuration Options
### Backend Configuration
| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `GROQ_API_KEY` | Required | Your Groq API key |
| `PORT` | `8000` | Server port |
| `ENVIRONMENT` | `development` | Deployment environment |
| `LOG_LEVEL` | `info` | Logging level |

### Desktop Widget Settings
- **Backend URL**: API endpoint (configurable in settings)
- **Microphone Selection**: Choose from available input devices
- **Chunk Duration**: Processing interval (default: 5 seconds)
- **Auto-Summarize**: Automatic summary generation (enabled by default)

## 📊 Performance & Monitoring
### Key Metrics
- **Response Time**: < 2 seconds for transcription
- **Accuracy**: 95%+ on clear audio
- **Memory Usage**: Optimized for long-running sessions
- **Connection Status**: Real-time backend health monitoring

### Monitoring Features
- **Chunk Processing Time**: Displayed in status bar
- **Connection Health**: Visual indicator in status bar
- **Error Handling**: Automatic retries with exponential backoff
- **Progress Indicators**: Visual feedback during processing

## 🔧 Troubleshooting
### Common Issues
1. **No Microphone Detected**
   - Ensure your microphone is properly connected
   - Check system audio settings
   - Try selecting a different microphone from the dropdown

2. **Backend Connection Failed**
   - Verify the backend is running
   - Check your internet connection
   - Confirm the backend URL is correct

3. **Poor Transcription Quality**
   - Ensure you're in a quiet environment
   - Check microphone positioning
   - Verify your microphone is selected in the dropdown

### Keyboard Shortcuts
- **Ctrl+R**: Toggle recording on/off
- **Ctrl+C**: Clear all content
- **F1**: Show help dialog

## 🚀 Production Deployment
### Backend Deployment
The FastAPI backend can be deployed to any cloud provider:
```bash
# Render Deployment
uvicorn app:app --host 0.0.0.0 --port $PORT

# Environment Variables
GROQ_API_KEY=gsk_your_key_here
ENVIRONMENT=production
PORT=8000
```

### Desktop Application Distribution
To distribute the desktop application:
1. Package using PyInstaller:
```bash
pip install pyinstaller
pyinstaller --onefile --windowed --icon=assets/icon.ico main.py
```
2. Create installers for your target platform
3. Distribute through your preferred channel


## 🤝 Contributing
We welcome contributions!

## 📞 Support
For support, please open an issue on our GitHub repository or contact our support team.

logger.info("🚀 Starting Nova.AI Backend v2.1.0...")
logger.error(f"❌ Groq initialization failed: {e}")
