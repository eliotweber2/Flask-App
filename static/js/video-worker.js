class VideoProcessor {
    constructor() {
        this.isProcessing = false;
        this.pollInterval = null;
        this.currentFilename = null;
    }

    // Start video processing
    async startProcessing(formData, filename) {
        this.isProcessing = true;
        this.currentFilename = filename;
        
        try {
            // Send initial processing request
            const response = await fetch('/interpreter', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            // Start polling for results
            this.startPolling();
            
            // Send status update to main thread
            self.postMessage({
                type: 'processing_started',
                filename: filename,
                message: 'Video processing started...'
            });

        } catch (error) {
            this.isProcessing = false;
            self.postMessage({
                type: 'error',
                message: `Failed to start processing: ${error.message}`
            });
        }
    }

    // Poll for processing results
    startPolling() {
        if (!this.isProcessing || !this.currentFilename) return;

        this.pollInterval = setInterval(async () => {
            try {
                const response = await fetch(`/check_result?filename=${encodeURIComponent(this.currentFilename)}`);
                const data = await response.json();

                if (data.ready) {
                    // Processing complete - get the final result
                    await this.getResult();
                } else {
                    // Still processing - send status update
                    self.postMessage({
                        type: 'processing_update',
                        message: 'Still processing... Please wait.'
                    });
                }
            } catch (error) {
                self.postMessage({
                    type: 'error',
                    message: `Polling error: ${error.message}`
                });
                this.stopProcessing();
            }
        }, 2000); // Poll every 2 seconds
    }

    // Get the final result
    async getResult() {
        try {
            const response = await fetch(`/get_result?filename=${encodeURIComponent(this.currentFilename)}`);
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            
            self.postMessage({
                type: 'processing_complete',
                result: result.text || 'No interpretation available',
                filename: this.currentFilename
            });

            this.stopProcessing();

        } catch (error) {
            self.postMessage({
                type: 'error',
                message: `Failed to get result: ${error.message}`
            });
            this.stopProcessing();
        }
    }

    // Stop processing and cleanup
    stopProcessing() {
        this.isProcessing = false;
        if (this.pollInterval) {
            clearInterval(this.pollInterval);
            this.pollInterval = null;
        }
        this.currentFilename = null;
    }
}

// Initialize processor
const processor = new VideoProcessor();

// Handle messages from main thread
self.onmessage = function(e) {
    const { type, data } = e.data;

    switch (type) {
        case 'start_processing':
            processor.startProcessing(data.formData, data.filename);
            break;
            
        case 'stop_processing':
            processor.stopProcessing();
            self.postMessage({
                type: 'processing_stopped',
                message: 'Processing stopped by user'
            });
            break;
            
        default:
            self.postMessage({
                type: 'error',
                message: `Unknown message type: ${type}`
            });
    }
};
