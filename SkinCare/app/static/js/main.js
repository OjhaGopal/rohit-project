// ── Nav: mobile menu toggle ──
const menuBar = document.querySelector('#menu-bar');
const nav     = document.querySelector('.nav');

menuBar.onclick = () => {
    menuBar.classList.toggle('fa-times');
    nav.classList.toggle('active');
};

// ── Nav: highlight active section on scroll ──
const sections  = document.querySelectorAll('section');
const navLinks  = document.querySelectorAll('header .nav a');

window.onscroll = () => {
    menuBar.classList.remove('fa-times');
    nav.classList.remove('active');

    sections.forEach(sec => {
        const top    = window.scrollY;
        const offset = sec.offsetTop - 150;
        const height = sec.offsetHeight;
        const id     = sec.getAttribute('id');

        if (top >= offset && top < offset + height) {
            navLinks.forEach(link => link.classList.remove('active'));
            const active = document.querySelector('header .nav a[href*=' + id + ']');
            if (active) active.classList.add('active');
        }
    });
};

// ── File upload: custom button ──
const realFileBtn = document.getElementById('real-file');
const customBtn   = document.getElementById('custom-button');
const customTxt   = document.getElementById('custom-text');

customBtn.addEventListener('click', () => realFileBtn.click());

realFileBtn.addEventListener('change', () => {
    if (realFileBtn.files && realFileBtn.files[0]) {
        customTxt.textContent = realFileBtn.files[0].name;
    } else {
        customTxt.textContent = 'No file chosen';
    }
});

// ── Camera ──
const cameraBtn      = document.getElementById('camera-button');
const cameraModal    = document.getElementById('camera-modal');
const cameraFeed     = document.getElementById('camera-feed');
const cameraCanvas   = document.getElementById('camera-canvas');
const cameraPreview  = document.getElementById('camera-preview');
const captureBtn     = document.getElementById('capture-btn');
const retakeBtn      = document.getElementById('retake-btn');
const usePhotoBtn    = document.getElementById('use-photo-btn');
const closeCameraBtn = document.getElementById('close-camera-btn');

let stream = null;

function startCamera() {
    cameraModal.style.display = 'flex';
    cameraFeed.hidden   = false;
    cameraPreview.hidden = true;
    captureBtn.hidden   = false;
    retakeBtn.hidden    = true;
    usePhotoBtn.hidden  = true;

    navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } })
        .then(s => {
            stream = s;
            cameraFeed.srcObject = s;
        })
        .catch(() => {
            // fallback to front camera if rear not available
            navigator.mediaDevices.getUserMedia({ video: true })
                .then(s => { stream = s; cameraFeed.srcObject = s; })
                .catch(() => {
                    alert('Camera access denied or not available.');
                    stopCamera();
                });
        });
}

function stopCamera() {
    if (stream) {
        stream.getTracks().forEach(t => t.stop());
        stream = null;
    }
    cameraModal.style.display = 'none';
}

cameraBtn.addEventListener('click', startCamera);
closeCameraBtn.addEventListener('click', stopCamera);

captureBtn.addEventListener('click', () => {
    cameraCanvas.width  = cameraFeed.videoWidth;
    cameraCanvas.height = cameraFeed.videoHeight;
    cameraCanvas.getContext('2d').drawImage(cameraFeed, 0, 0);

    cameraPreview.src    = cameraCanvas.toDataURL('image/jpeg');
    cameraFeed.hidden    = true;
    cameraPreview.hidden = false;
    captureBtn.hidden    = true;
    retakeBtn.hidden     = false;
    usePhotoBtn.hidden   = false;
});

retakeBtn.addEventListener('click', () => {
    cameraFeed.hidden    = false;
    cameraPreview.hidden = true;
    captureBtn.hidden    = false;
    retakeBtn.hidden     = true;
    usePhotoBtn.hidden   = true;
});

usePhotoBtn.addEventListener('click', () => {
    cameraCanvas.toBlob(blob => {
        const file = new File([blob], 'camera-capture.jpg', { type: 'image/jpeg' });
        const dt   = new DataTransfer();
        dt.items.add(file);
        realFileBtn.files = dt.files;
        customTxt.textContent = 'camera-capture.jpg';
        stopCamera();
    }, 'image/jpeg', 0.92);
});
