document.addEventListener('DOMContentLoaded', function() {
    const fileInput = document.getElementById('video');
    const fileLabel = document.querySelector('label[for="video"]');
    if (fileInput && fileLabel) {
        fileInput.addEventListener('change', function(e) {
            const fileName = e.target.files[0] ? e.target.files[0].name : 'Choose Video';
            fileLabel.textContent = fileName;
        });
    }
});