 recordingIndicator.textContent = 'Interpreting...';
        resultsArea.innerHTML = '';
        resultsArea.appendChild(recordingIndicator);
        console.log('Started recording');
        setTimeout(() => {
            const text = document.createElement('div');
            text.className = 'interpretation-text';
            text.textContent = 'Hello, how are you today?';
            resultsArea.appendChild(text);
        }, 2000);
        
        setTimeout(() => {
            const text = document.createElement('div');
            text.className = 'interpretation-text';
            text.textContent = 'I am learning sign language.';
            resultsArea.appendChild(text);
        }, 4000);
    });
    stopRecordingButton.addEventListener('click', function() {
        // In a real app, this would stop the recording
        startRecordingButton.disabled = false;
        stopRecordingButton.disabled = true;
        const text = document.createElement('div');
        text.className = 'interpretation-text';
        text.style.marginTop = '10px';
        text.textContent = 'Thank you for using our interpreter!';
        resultsArea.appendChild(text);
        
        console.log('Stopped recording');
    });
});
