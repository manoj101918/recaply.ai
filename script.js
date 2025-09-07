// Nova.AI Desktop Interface JavaScript
class NovaAI {
    constructor() {
        this.apiBaseUrl = 'http://localhost:8000';
        this.isRecording = false;
        this.mediaRecorder = null;
        this.audioChunks = [];
        this.recordingStartTime = null;
        this.recordingInterval = null;
        this.sessionStartTime = Date.now();
        this.stats = {
            chunksProcessed: 0,
            totalWords: 0,
            sessionTime: 0
        };
        
        this.initialize();
    }

    async initialize() {
        this.showLoading('Initializing Nova.AI...');
        
        try {
            // Setup event listeners
            this.setupEventListeners();
            
            // Initialize microphones
            await this.initializeMicrophones();
            
            // Check backend health
            await this.checkBackendHealth();
            
            // Start periodic health checks
            this.startHealthMonitoring();
            
            // Start session timer
            this.startSessionTimer();
            
            this.hideLoading();
            this.showToast('success', 'Nova.AI Ready', 'System initialized successfully');
            
        } catch (error) {
            console.error('Initialization error:', error);
            this.hideLoading();
            this.showToast('error', 'Initialization Failed', error.message);
        }
    }

    setupEventListeners() {
        // Recording controls
        document.getElementById('start-recording').addEventListener('click', () => this.startRecording());
        document.getElementById('stop-recording').addEventListener('click', () => this.stopRecording());
        document.getElementById('transcribe-file').addEventListener('click', () => this.transcribeFile());
        
        // File upload
        const fileInput = document.getElementById('audio-file');
        fileInput.addEventListener('change', this.handleFileSelect.bind(this));
        
        // Control buttons
        document.getElementById('clear-transcript').addEventListener('click', () => this.clearTranscript());
        document.getElementById('export-transcript').addEventListener('click', () => this.exportTranscript());
        document.getElementById('generate-summary').addEventListener('click', () => this.generateSummary());
        document.getElementById('get-suggestions').addEventListener('click', () => this.getSuggestions());
        document.getElementById('refresh-health').addEventListener('click', () => this.checkBackendHealth());
        
        // AI tabs
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', (e) => this.switchTab(e.target.dataset.tab));
        });
        
        // Keyboard shortcuts
        document.addEventListener('keydown', this.handleKeyboardShortcuts.bind(this));
        
        // Toast close buttons
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('toast-close')) {
                this.closeToast(e.target.closest('.toast'));
            }
        });
    }

    async initializeMicrophones() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            const devices = await navigator.mediaDevices.enumerateDevices();
            const audioDevices = devices.filter(device => device.kind === 'audioinput');
            
            const micSelect = document.getElementById('microphone-select');
            micSelect.innerHTML = '';
            
            if (audioDevices.length === 0) {
                micSelect.innerHTML = '<option value="">No microphones detected</option>';
                return;
            }
            
            audioDevices.forEach((device, index) => {
                const option = document.createElement('option');
                option.value = device.deviceId;
                option.textContent = device.label || `Microphone ${index + 1}`;
                micSelect.appendChild(option);
            });
            
            // Stop the initial stream
            stream.getTracks().forEach(track => track.stop());
            
            this.updateConnectionStatus('connected', 'Microphones detected');
            
        } catch (error) {
            console.error('Microphone initialization error:', error);
            this.updateConnectionStatus('error', 'Microphone access denied');
            document.getElementById('microphone-select').innerHTML = 
                '<option value="">Microphone access required</option>';
        }
    }

    async checkBackendHealth() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/health`);
            const health = await response.json();
            
            // Update connection status
            if (health.status === 'healthy') {
                this.updateConnectionStatus('connected', 'Connected to backend');
                this.updateOpenAIStatus('connected', 'OpenAI Ready');
            } else {
                this.updateConnectionStatus('warning', `Backend: ${health.status}`);
                this.updateOpenAIStatus('warning', health.message);
            }
            
            // Update health display
            this.updateHealthDisplay(health);
            
            return health;
            
        } catch (error) {
            console.error('Health check error:', error);
            this.updateConnectionStatus('error', 'Backend offline');
            this.updateOpenAIStatus('error', 'Service unavailable');
            
            // Update health display with error state
            this.updateHealthDisplay({
                status: 'error',
                message: 'Backend unreachable',
                uptime: 0,
                models_loaded: {}
            });
            
            throw error;
        }
    }

    updateConnectionStatus(status, message) {
        const statusElement = document.getElementById('connection-status');
        const icon = statusElement.querySelector('i');
        const text = statusElement.querySelector('span');
        
        // Remove existing status classes
        statusElement.classList.remove('connected', 'error', 'warning');
        statusElement.classList.add(status);
        
        text.textContent = message;
    }

    updateOpenAIStatus(status, message) {
        const statusElement = document.getElementById('openai-status');
        const icon = statusElement.querySelector('i');
        const text = statusElement.querySelector('span');
        
        statusElement.classList.remove('connected', 'error', 'warning');
        statusElement.classList.add(status);
        
        text.textContent = message;
    }

    updateHealthDisplay(health) {
        // Backend status
        const backendHealth = document.getElementById('backend-health');
        backendHealth.textContent = health.status.charAt(0).toUpperCase() + health.status.slice(1);
        backendHealth.className = `health-status ${health.status === 'healthy' ? '' : health.status}`;
        
        // Models status
        const modelsHealth = document.getElementById('models-health');
        const modelsLoaded = health.models_loaded || {};
        const loadedCount = Object.values(modelsLoaded).filter(Boolean).length;
        const totalCount = Object.keys(modelsLoaded).length;
        
        if (totalCount > 0) {
            modelsHealth.textContent = `${loadedCount}/${totalCount} Ready`;
            modelsHealth.className = `health-status ${loadedCount === totalCount ? '' : 'warning'}`;
        } else {
            modelsHealth.textContent = 'Unknown';
            modelsHealth.className = 'health-status warning';
        }
        
        // Uptime
        const uptimeDisplay = document.getElementById('uptime-display');
        uptimeDisplay.textContent = this.formatUptime(health.uptime || 0);
        
        // Performance (simple heuristic based on health status)
        const performanceDisplay = document.getElementById('performance-display');
        if (health.status === 'healthy') {
            performanceDisplay.textContent = 'Excellent';
            performanceDisplay.className = 'health-status';
        } else if (health.status === 'degraded') {
            performanceDisplay.textContent = 'Good';
            performanceDisplay.className = 'health-status warning';
        } else {
            performanceDisplay.textContent = 'Poor';
            performanceDisplay.className = 'health-status error';
        }
    }

    formatUptime(seconds) {
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        const secs = Math.floor(seconds % 60);
        
        return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    }

    startHealthMonitoring() {
        // Check health every 30 seconds
        setInterval(() => {
            this.checkBackendHealth().catch(() => {
                // Silent fail for periodic checks
            });
        }, 30000);
    }

    startSessionTimer() {
        setInterval(() => {
            const elapsed = Math.floor((Date.now() - this.sessionStartTime) / 1000);
            document.getElementById('session-time').textContent = this.formatUptime(elapsed);
        }, 1000);
    }

    async startRecording() {
        try {
            const micSelect = document.getElementById('microphone-select');
            const deviceId = micSelect.value;
            
            if (!deviceId) {
                this.showToast('warning', 'No Microphone', 'Please select a microphone first');
                return;
            }
            
            const constraints = {
                audio: {
                    deviceId: deviceId ? { exact: deviceId } : undefined,
                    sampleRate: 16000,
                    channelCount: 1,
                    echoCancellation: true,
                    noiseSuppression: true
                }
            };
            
            const stream = await navigator.mediaDevices.getUserMedia(constraints);
            
            this.mediaRecorder = new MediaRecorder(stream, {
                mimeType: 'audio/webm;codecs=opus'
            });
            
            this.audioChunks = [];
            this.recordingStartTime = Date.now();
            
            this.mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    this.audioChunks.push(event.data);
                }
            };
            
            this.mediaRecorder.onstop = () => {
                this.processRecording();
            };
            
            // Start recording
            this.mediaRecorder.start(5000); // Collect data every 5 seconds
            this.isRecording = true;
            
            // Update UI
            this.updateRecordingUI(true);
            this.startRecordingTimer();
            
            this.showToast('success', 'Recording Started', 'Capturing audio...');
            
        } catch (error) {
            console.error('Recording start error:', error);
            this.showToast('error', 'Recording Failed', error.message);
        }
    }

    stopRecording() {
        if (this.mediaRecorder && this.isRecording) {
            this.mediaRecorder.stop();
            this.mediaRecorder.stream.getTracks().forEach(track => track.stop());
            this.isRecording = false;
            
            this.updateRecordingUI(false);
            this.stopRecordingTimer();
            
            this.showToast('success', 'Recording Stopped', 'Processing audio...');
        }
    }

    updateRecordingUI(recording) {
        const startBtn = document.getElementById('start-recording');
        const stopBtn = document.getElementById('stop-recording');
        const statusIndicator = document.querySelector('.status-indicator');
        const statusText = statusIndicator.querySelector('span');
        
        if (recording) {
            startBtn.disabled = true;
            stopBtn.disabled = false;
            statusIndicator.classList.add('recording');
            statusText.textContent = 'Recording...';
        } else {
            startBtn.disabled = false;
            stopBtn.disabled = true;
            statusIndicator.classList.remove('recording');
            statusText.textContent = 'Ready';
        }
    }

    startRecordingTimer() {
        this.recordingInterval = setInterval(() => {
            if (this.recordingStartTime) {
                const elapsed = Math.floor((Date.now() - this.recordingStartTime) / 1000);
                const minutes = Math.floor(elapsed / 60);
                const seconds = elapsed % 60;
                document.getElementById('recording-timer').textContent = 
                    `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
            }
        }, 1000);
    }

    stopRecordingTimer() {
        if (this.recordingInterval) {
            clearInterval(this.recordingInterval);
            this.recordingInterval = null;
        }
        document.getElementById('recording-timer').textContent = '00:00';
    }

    async processRecording() {
        if (this.audioChunks.length === 0) return;
        
        try {
            const audioBlob = new Blob(this.audioChunks, { type: 'audio/webm' });
            await this.transcribeAudio(audioBlob);
            
        } catch (error) {
            console.error('Recording processing error:', error);
            this.showToast('error', 'Processing Failed', error.message);
        }
    }

    handleFileSelect(event) {
        const file = event.target.files[0];
        const fileNameSpan = document.getElementById('file-name');
        const transcribeBtn = document.getElementById('transcribe-file');
        
        if (file) {
            fileNameSpan.textContent = `${file.name} (${this.formatFileSize(file.size)})`;
            transcribeBtn.disabled = false;
        } else {
            fileNameSpan.textContent = 'No file chosen';
            transcribeBtn.disabled = true;
        }
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    async transcribeFile() {
        const fileInput = document.getElementById('audio-file');
        const file = fileInput.files[0];
        
        if (!file) {
            this.showToast('warning', 'No File', 'Please select an audio file first');
            return;
        }
        
        await this.transcribeAudio(file);
    }

    async transcribeAudio(audioBlob) {
        this.showLoading('Transcribing audio...');
        
        try {
            const formData = new FormData();
            formData.append('audio', audioBlob, 'audio.webm');
            
            const response = await fetch(`${this.apiBaseUrl}/transcribe`, {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Transcription failed');
            }
            
            const result = await response.json();
            this.displayTranscription(result);
            this.updateTranscriptionStats(result);
            
            // Enable AI features
            this.enableAIFeatures();
            
            this.hideLoading();
            this.showToast('success', 'Transcription Complete', 
                `Processed in ${result.processing_time.toFixed(2)}s`);
            
        } catch (error) {
            console.error('Transcription error:', error);
            this.hideLoading();
            this.showToast('error', 'Transcription Failed', error.message);
        }
    }

    displayTranscription(result) {
        const transcriptDisplay = document.getElementById('transcript-display');
        
        // Remove placeholder if present
        const placeholder = transcriptDisplay.querySelector('.placeholder');
        if (placeholder) {
            placeholder.remove();
        }
        
        // Create transcript chunk
        const chunk = document.createElement('div');
        chunk.className = 'transcript-chunk';
        
        const timestamp = document.createElement('div');
        timestamp.className = 'chunk-timestamp';
        timestamp.textContent = new Date().toLocaleTimeString();
        
        const text = document.createElement('div');
        text.className = 'chunk-text';
        text.textContent = result.text || '[No text detected]';
        
        chunk.appendChild(timestamp);
        chunk.appendChild(text);
        transcriptDisplay.appendChild(chunk);
        
        // Auto-scroll to bottom
        transcriptDisplay.scrollTop = transcriptDisplay.scrollHeight;
        
        // Update stats
        this.stats.chunksProcessed++;
        this.stats.totalWords += result.text ? result.text.split(' ').length : 0;
        
        this.updateStatsDisplay();
    }

    updateTranscriptionStats(result) {
        // Word count
        const wordCount = result.text ? result.text.split(' ').length : 0;
        document.getElementById('word-count').textContent = wordCount;
        
        // Processing time
        document.getElementById('processing-time').textContent = `${result.processing_time.toFixed(2)}s`;
        
        // Confidence
        const confidence = Math.round((result.confidence || 0) * 100);
        const confidenceFill = document.getElementById('confidence-fill');
        const confidenceText = document.getElementById('confidence-text');
        
        confidenceFill.style.width = `${confidence}%`;
        confidenceText.textContent = `${confidence}%`;
    }

    updateStatsDisplay() {
        document.getElementById('chunks-processed').textContent = this.stats.chunksProcessed;
        document.getElementById('total-words').textContent = this.stats.totalWords;
    }

    enableAIFeatures() {
        document.getElementById('generate-summary').disabled = false;
        document.getElementById('get-suggestions').disabled = false;
    }

    async generateSummary() {
        const transcriptDisplay = document.getElementById('transcript-display');
        const transcriptText = this.extractTranscriptText(transcriptDisplay);
        
        if (!transcriptText || transcriptText.length < 50) {
            this.showToast('warning', 'Insufficient Content', 'Need more text for summary generation');
            return;
        }
        
        this.showLoading('Generating AI summary...');
        
        try {
            const response = await fetch(`${this.apiBaseUrl}/summarize`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    text: transcriptText
                })
            });
            
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Summary generation failed');
            }
            
            const result = await response.json();
            this.displaySummary(result);
            
            this.hideLoading();
            this.showToast('success', 'Summary Generated', 
                `Processed in ${result.processing_time.toFixed(2)}s`);
            
        } catch (error) {
            console.error('Summary generation error:', error);
            this.hideLoading();
            this.showToast('error', 'Summary Failed', error.message);
        }
    }

    displaySummary(result) {
        const summaryDisplay = document.getElementById('summary-display');
        const summarySections = document.getElementById('summary-sections');
        
        // Remove placeholder
        const placeholder = summaryDisplay.querySelector('.placeholder');
        if (placeholder) {
            placeholder.remove();
        }
        
        // Show summary sections
        summarySections.style.display = 'flex';
        
        // Update overview
        const summaryOverview = document.getElementById('summary-overview');
        summaryOverview.textContent = result.summary || 'No summary available';
        
        // Update key points
        const keyPointsList = document.getElementById('key-points-list');
        keyPointsList.innerHTML = '';
        
        if (result.key_points && result.key_points.length > 0) {
            result.key_points.forEach(point => {
                const li = document.createElement('li');
                li.textContent = point;
                keyPointsList.appendChild(li);
            });
        } else {
            const li = document.createElement('li');
            li.textContent = 'No key points identified';
            li.style.fontStyle = 'italic';
            li.style.color = 'var(--text-muted)';
            keyPointsList.appendChild(li);
        }
        
        // Update action items
        const actionItemsList = document.getElementById('action-items-list');
        actionItemsList.innerHTML = '';
        
        if (result.action_items && result.action_items.length > 0) {
            result.action_items.forEach(item => {
                const li = document.createElement('li');
                li.textContent = item;
                actionItemsList.appendChild(li);
            });
        } else {
            const li = document.createElement('li');
            li.textContent = 'No action items identified';
            li.style.fontStyle = 'italic';
            li.style.color = 'var(--text-muted)';
            actionItemsList.appendChild(li);
        }
    }

    async getSuggestions() {
        const transcriptDisplay = document.getElementById('transcript-display');
        const transcriptText = this.extractTranscriptText(transcriptDisplay);
        
        if (!transcriptText || transcriptText.length < 20) {
            this.showToast('warning', 'Insufficient Content', 'Need more context for suggestions');
            return;
        }
        
        this.showLoading('Generating response suggestions...');
        
        try {
            const response = await fetch(`${this.apiBaseUrl}/suggest_response`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    text: transcriptText
                })
            });
            
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Suggestion generation failed');
            }
            
            const result = await response.json();
            this.displaySuggestions(result);
            
            this.hideLoading();
            this.showToast('success', 'Suggestions Ready', 
                `Generated in ${result.processing_time.toFixed(2)}s`);
            
        } catch (error) {
            console.error('Suggestions error:', error);
            this.hideLoading();
            this.showToast('error', 'Suggestions Failed', error.message);
        }
    }

    displaySuggestions(result) {
        const suggestionsDisplay = document.getElementById('suggestions-display');
        
        // Remove placeholder
        const placeholder = suggestionsDisplay.querySelector('.placeholder');
        if (placeholder) {
            placeholder.remove();
        }
        
        // Create suggestion item
        const suggestionItem = document.createElement('div');
        suggestionItem.className = 'suggestion-item';
        
        const context = document.createElement('div');
        context.className = 'suggestion-context';
        context.textContent = `Context: ${result.context_identified || 'General response'}`;
        
        const suggestion = document.createElement('div');
        suggestion.className = 'suggestion-text';
        suggestion.textContent = result.suggestion || 'No suggestion available';
        
        const confidence = document.createElement('div');
        confidence.className = 'suggestion-confidence';
        confidence.textContent = `Confidence: ${result.confidence || 'Medium'}`;
        
        suggestionItem.appendChild(context);
        suggestionItem.appendChild(suggestion);
        suggestionItem.appendChild(confidence);
        
        // Clear previous suggestions and add new one
        suggestionsDisplay.innerHTML = '';
        suggestionsDisplay.appendChild(suggestionItem);
    }

    extractTranscriptText(transcriptDisplay) {
        const chunks = transcriptDisplay.querySelectorAll('.chunk-text');
        return Array.from(chunks).map(chunk => chunk.textContent).join(' ');
    }

    switchTab(tabName) {
        // Update tab buttons
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');
        
        // Update tab content
        document.querySelectorAll('.ai-content').forEach(content => {
            content.style.display = 'none';
        });
        document.getElementById(`${tabName}-tab`).style.display = 'block';
    }

    clearTranscript() {
        const transcriptDisplay = document.getElementById('transcript-display');
        transcriptDisplay.innerHTML = `
            <div class="placeholder">
                <i class="fas fa-microphone-alt"></i>
                <p>Start recording or upload an audio file to begin transcription</p>
            </div>
        `;
        
        // Reset stats
        this.stats.chunksProcessed = 0;
        this.stats.totalWords = 0;
        this.updateStatsDisplay();
        
        // Reset transcription stats display
        document.getElementById('word-count').textContent = '0';
        document.getElementById('processing-time').textContent = '0.00s';
        document.getElementById('confidence-fill').style.width = '0%';
        document.getElementById('confidence-text').textContent = '0%';
        
        // Clear AI sections
        this.clearAISections();
        
        // Disable AI features
        document.getElementById('generate-summary').disabled = true;
        document.getElementById('get-suggestions').disabled = true;
        
        this.showToast('success', 'Cleared', 'Transcript cleared successfully');
    }

    clearAISections() {
        // Clear summary
        const summaryDisplay = document.getElementById('summary-display');
        const summarySections = document.getElementById('summary-sections');
        
        summaryDisplay.innerHTML = `
            <div class="placeholder">
                <i class="fas fa-magic"></i>
                <p>AI-powered summary will appear here after transcription</p>
            </div>
        `;
        summarySections.style.display = 'none';
        
        // Clear suggestions
        const suggestionsDisplay = document.getElementById('suggestions-display');
        suggestionsDisplay.innerHTML = `
            <div class="placeholder">
                <i class="fas fa-comments"></i>
                <p>Smart response suggestions will appear here</p>
            </div>
        `;
    }

    exportTranscript() {
        const transcriptDisplay = document.getElementById('transcript-display');
        const transcriptText = this.extractTranscriptText(transcriptDisplay);
        
        if (!transcriptText) {
            this.showToast('warning', 'No Content', 'No transcript available to export');
            return;
        }
        
        // Get additional info
        const chunks = transcriptDisplay.querySelectorAll('.transcript-chunk');
        let exportContent = `Nova.AI Transcript Export\n`;
        exportContent += `Generated: ${new Date().toLocaleString()}\n`;
        exportContent += `Total Chunks: ${chunks.length}\n`;
        exportContent += `Total Words: ${this.stats.totalWords}\n`;
        exportContent += `\n${'='.repeat(50)}\n\n`;
        
        // Add timestamped content
        chunks.forEach(chunk => {
            const timestamp = chunk.querySelector('.chunk-timestamp').textContent;
            const text = chunk.querySelector('.chunk-text').textContent;
            exportContent += `[${timestamp}] ${text}\n\n`;
        });
        
        // Create and download file
        const blob = new Blob([exportContent], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `nova-ai-transcript-${new Date().toISOString().split('T')[0]}.txt`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        
        this.showToast('success', 'Export Complete', 'Transcript saved to downloads');
    }

    handleKeyboardShortcuts(event) {
        if (event.ctrlKey || event.metaKey) {
            switch (event.key.toLowerCase()) {
                case 'r':
                    event.preventDefault();
                    if (this.isRecording) {
                        this.stopRecording();
                    } else {
                        this.startRecording();
                    }
                    break;
                case 'c':
                    if (event.shiftKey) {
                        event.preventDefault();
                        this.clearTranscript();
                    }
                    break;
                case 's':
                    event.preventDefault();
                    this.generateSummary();
                    break;
                case 'e':
                    event.preventDefault();
                    this.exportTranscript();
                    break;
            }
        }
        
        if (event.key === 'F1') {
            event.preventDefault();
            this.showHelpDialog();
        }
    }

    showHelpDialog() {
        const helpContent = `
            Nova.AI Keyboard Shortcuts:
            
            Ctrl+R - Toggle Recording
            Ctrl+Shift+C - Clear Transcript
            Ctrl+S - Generate Summary
            Ctrl+E - Export Transcript
            F1 - Show Help
        `;
        
        this.showToast('info', 'Keyboard Shortcuts', helpContent);
    }

    showLoading(message = 'Processing...') {
        const overlay = document.getElementById('loading-overlay');
        const messageElement = document.getElementById('loading-message');
        
        messageElement.textContent = message;
        overlay.classList.add('show');
    }

    hideLoading() {
        const overlay = document.getElementById('loading-overlay');
        overlay.classList.remove('show');
    }

    showToast(type = 'info', title, message, duration = 5000) {
        const container = document.getElementById('toast-container');
        
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        
        const iconMap = {
            success: 'fas fa-check-circle',
            error: 'fas fa-exclamation-circle',
            warning: 'fas fa-exclamation-triangle',
            info: 'fas fa-info-circle'
        };
        
        toast.innerHTML = `
            <div class="toast-icon">
                <i class="${iconMap[type] || iconMap.info}"></i>
            </div>
            <div class="toast-content">
                <div class="toast-title">${title}</div>
                <div class="toast-message">${message}</div>
            </div>
            <button class="toast-close">
                <i class="fas fa-times"></i>
            </button>
        `;
        
        container.appendChild(toast);
        
        // Auto remove after duration
        setTimeout(() => {
            this.closeToast(toast);
        }, duration);
    }

    closeToast(toast) {
        toast.style.animation = 'slideOutRight 0.3s ease-out forwards';
        setTimeout(() => {
            if (toast.parentNode) {
                toast.parentNode.removeChild(toast);
            }
        }, 300);
    }

    // Utility method for API configuration
    setApiBaseUrl(url) {
        this.apiBaseUrl = url;
        this.showToast('info', 'API Updated', `Backend URL set to ${url}`);
    }
}

// Add slideOutRight animation
const style = document.createElement('style');
style.textContent = `
    @keyframes slideOutRight {
        0% {
            opacity: 1;
            transform: translateX(0);
        }
        100% {
            opacity: 0;
            transform: translateX(100%);
        }
    }
`;
document.head.appendChild(style);

// Initialize Nova.AI when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.novaAI = new NovaAI();
});

// Handle page visibility changes
document.addEventListener('visibilitychange', () => {
    if (document.hidden && window.novaAI && window.novaAI.isRecording) {
        // Optionally pause recording when page is hidden
        console.log('Page hidden while recording - consider pausing');
    }
});

// Handle beforeunload to warn about active recording
window.addEventListener('beforeunload', (event) => {
    if (window.novaAI && window.novaAI.isRecording) {
        event.preventDefault();
        event.returnValue = 'Recording is in progress. Are you sure you want to leave?';
        return event.returnValue;
    }
});

// Export for potential external use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = NovaAI;
}