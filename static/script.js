const video = document.getElementById('webcam');
const emotionText = document.getElementById('emotion-text');
const confidenceFill = document.getElementById('confidence-fill');
const confidenceText = document.getElementById('confidence-text');
const faceBox = document.getElementById('face-box');
const toggleBtn = document.getElementById('toggle-cam');
const tabBtns = document.querySelectorAll('.tab-btn');
const tabContents = document.querySelectorAll('.tab-content');
const fileUpload = document.getElementById('file-upload');
const imagePreview = document.getElementById('image-preview');
const imagePreviewContainer = document.getElementById('image-preview-container');
const dropzone = document.getElementById('dropzone');
const clearBtn = document.getElementById('clear-image');
const placeholder = document.getElementById('camera-placeholder');

let isStreaming = false;
let stream = null;
let predictionInterval = null;

async function startCamera() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({ 
            video: { 
                width: { ideal: 1280 },
                height: { ideal: 720 },
                facingMode: "user" 
            } 
        });
        video.srcObject = stream;
        video.style.display = 'block';
        placeholder.style.display = 'none';
        isStreaming = true;
        toggleBtn.innerHTML = '<span>Stop Camera</span>';
        
        // Start processing frames
        startPredicting();
    } catch (err) {
        console.error("Error accessing webcam:", err);
        emotionText.innerText = "Camera Error";
    }
}

function stopCamera() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        video.srcObject = null;
        video.style.display = 'none';
        placeholder.style.display = 'block';
        isStreaming = false;
        toggleBtn.innerHTML = '<span>Start Camera</span>';
        clearInterval(predictionInterval);
        resetUI();
    }
}

function startPredicting() {
    predictionInterval = setInterval(async () => {
        if (!isStreaming) return;

        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext('2d');
        ctx.translate(canvas.width, 0);
        ctx.scale(-1, 1);
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        canvas.toBlob(async (blob) => {
            const formData = new FormData();
            formData.append('image', blob);

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });
                const data = await response.json();

                // Only update if we are still on the webcam tab and streaming
                const activeTab = document.querySelector('.tab-btn.active').dataset.tab;
                if (data.emotion && isStreaming && activeTab === 'webcam-tab') {
                    updateUI(data);
                }
            } catch (err) {
                console.error("Prediction error:", err);
            }
        }, 'image/jpeg');
    }, 200); // Predict 5 times per second
}

function updateUI(data) {
    emotionText.innerText = data.emotion;
    confidenceFill.style.width = `${data.confidence}%`;
    confidenceText.innerText = `${data.confidence}%`;

    if (data.box && isStreaming) {
        const [x, y, w, h] = data.box;
        const videoWidth = video.offsetWidth;
        const videoHeight = video.offsetHeight;
        const actualWidth = video.videoWidth;
        const actualHeight = video.videoHeight;

        // Calculate scale
        const scaleX = videoWidth / actualWidth;
        const scaleY = videoHeight / actualHeight;

        faceBox.style.display = 'block';
        // Adjust for mirrored video
        faceBox.style.left = `${videoWidth - (x + w) * scaleX}px`;
        faceBox.style.top = `${y * scaleY}px`;
        faceBox.style.width = `${w * scaleX}px`;
        faceBox.style.height = `${h * scaleY}px`;
    } else {
        faceBox.style.display = 'none';
    }
}

// Tab Switching Logic
tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        const target = btn.dataset.tab;
        resetUI(); // Clear old results
        
        // Update Buttons
        tabBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        
        // Update Content
        tabContents.forEach(content => {
            content.classList.remove('active');
            if(content.id === target) content.classList.add('active');
        });

        // Hide info panel on About tab
        const infoPanel = document.querySelector('.info-panel');
        const mainContent = document.querySelector('.main-content');
        if (target === 'about-tab') {
            infoPanel.style.display = 'none';
            mainContent.style.gridTemplateColumns = '1fr';
        } else {
            infoPanel.style.display = 'flex';
            mainContent.style.gridTemplateColumns = '';
        }

        // Specific actions for tabs
        if (target === 'webcam-tab') {
            document.querySelector('.controls').style.display = 'flex';
            // Keep camera off per user preference until "Start Camera" is clicked
        } else {
            document.querySelector('.controls').style.display = 'none';
            stopCamera(); // Ensure it stops
        }
    });
});

function resetUI() {
    emotionText.innerText = "Scanning...";
    confidenceFill.style.width = "0%";
    confidenceText.innerText = "0%";
    faceBox.style.display = 'none';
}

function clearUpload() {
    fileUpload.value = '';
    imagePreview.src = '';
    imagePreviewContainer.style.display = 'none';
    dropzone.style.display = 'flex';
    resetUI();
}

clearBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    clearUpload();
});

// File Upload Prediction
fileUpload.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    // Show Preview
    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.src = e.target.result;
        imagePreviewContainer.style.display = 'flex';
        dropzone.style.display = 'none';
    };
    reader.readAsDataURL(file);

    // Predict
    const formData = new FormData();
    formData.append('image', file);

    emotionText.innerText = "Analyzing...";
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error('Prediction failed');
        }
        
        const data = await response.json();
        
        if (data.error) {
            emotionText.innerText = "Model could not predict";
            confidenceFill.style.width = "0%";
            confidenceText.innerText = "0%";
        } else if (data.emotion === 'No Face Detected') {
            emotionText.innerText = "Model could not predict";
            confidenceFill.style.width = "0%";
            confidenceText.innerText = "0%";
        } else if (data.emotion) {
            updateUI(data);
        } else {
            emotionText.innerText = "Model could not predict";
        }
    } catch (err) {
        console.error("Upload prediction error:", err);
        emotionText.innerText = "Model could not predict";
        confidenceFill.style.width = "0%";
        confidenceText.innerText = "0%";
    }
});

// Click on preview to re-upload
imagePreviewContainer.addEventListener('click', (e) => {
    if (e.target !== clearBtn) {
        fileUpload.click();
    }
});

toggleBtn.addEventListener('click', () => {
    if (isStreaming) {
        stopCamera();
    } else {
        startCamera();
    }
});

// Removed initial startCamera() to respect user privacy
