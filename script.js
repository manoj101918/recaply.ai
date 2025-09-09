class NovaAI {
    constructor() {
        this.apiBaseUrl = "https://recaply-ai.onrender.com";
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
            this.setupEventListeners();
            await this.initializeMicrophones();
            await this.checkBackendHealth();
            this.startHealthMonitoring();
            this.startSessionTimer();

            this.hideLoading();
            this.showToast('success', 'Nova.AI Ready', 'System initialized successfully');
        } catch (error) {
            console.error('Initialization error:', error);
            this.hideLoading();
            this.showToast('error', 'Initialization Failed', error.message);
        }
    }

    // ✅ FIXED
    async checkBackendHealth() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/health`);
            const health = await response.json();

            if (health.status === 'healthy') {
                this.updateConnectionStatus('connected', 'Connected to backend');
                this.updateOpenAIStatus('connected', 'OpenAI Ready');
            } else {
                this.updateConnectionStatus('warning', "Backend: " + health.status);
                this.updateOpenAIStatus('warning', health.message);
            }

            this.updateHealthDisplay(health);
            return health;

        } catch (error) {
            console.error('Health check error:', error);
            this.updateConnectionStatus('error', 'Backend offline');
            this.updateOpenAIStatus('error', 'Service unavailable');

            this.updateHealthDisplay({
                status: 'error',
                message: 'Backend unreachable',
                uptime: 0,
                models_loaded: {}
            });

            throw error;
        }
    }

    // ✅ FIXED
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

    // ✅ FIXED
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
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: transcriptText })
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

    // ✅ FIXED
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
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: transcriptText })
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
}
