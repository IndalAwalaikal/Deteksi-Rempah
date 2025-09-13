// Enhanced Spice Detection App - Complete JavaScript (Cleaned & Deduplicated)

// DOM Elements
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const browseBtn = document.getElementById('browseBtn');
const previewSection = document.getElementById('previewSection');
const previewImg = document.getElementById('previewImg');
const processingSection = document.getElementById('processingSection');
const resultSection = document.getElementById('resultSection');
const loading = document.getElementById('loading');
const classNameEl = document.getElementById('className');
const confidenceEl = document.getElementById('confidence');
const chartCanvas = document.getElementById('chart');

// Navigation elements
const hamburger = document.querySelector('.hamburger');
const navMenu = document.querySelector('.nav-menu');
const navLinks = document.querySelectorAll('.nav-link');

let currentChart = null;
let isProcessing = false;

// Initialize app
function init() {
    setupEventListeners();
    setupNavigation();
    addScrollEffects();
    addScrollToTopButton();
    addLoadingEffects();
}

// Setup navigation functionality
function setupNavigation() {
    if (hamburger && navMenu) {
        hamburger.addEventListener('click', () => {
            hamburger.classList.toggle('active');
            navMenu.classList.toggle('active');
        });
    }

    // Smooth scrolling for navigation links
    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            // Close mobile menu if open
            if (hamburger) hamburger.classList.remove('active');
            if (navMenu) navMenu.classList.remove('active');
            // Get target section
            const targetId = link.getAttribute('href');
            const targetSection = document.querySelector(targetId);
            if (targetSection) {
                const header = document.querySelector('.header');
                const headerHeight = header ? header.offsetHeight : 80;
                const targetPosition = targetSection.offsetTop - headerHeight - 20;
                window.scrollTo({
                    top: targetPosition,
                    behavior: 'smooth'
                });
            }
        });
    });

    // Update active nav item on scroll
    window.addEventListener('scroll', updateActiveNavItem);
}

// Update active navigation item based on scroll position
function updateActiveNavItem() {
    const sections = document.querySelectorAll('section[id]');
    const header = document.querySelector('.header');
    const headerHeight = header ? header.offsetHeight : 80;
    let currentSection = '';

    sections.forEach(section => {
        const sectionTop = section.offsetTop - headerHeight - 100;
        const sectionHeight = section.offsetHeight;
        if (window.scrollY >= sectionTop && window.scrollY < sectionTop + sectionHeight) {
            currentSection = section.getAttribute('id');
        }
    });

    navLinks.forEach(link => {
        link.classList.remove('active');
        if (link.getAttribute('href') === `#${currentSection}`) {
            link.classList.add('active');
        }
    });
}

// Drag & Drop Functions
function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

function highlight() {
    dropZone.classList.add('highlight');
}

function unhighlight() {
    dropZone.classList.remove('highlight');
}

function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;
    handleFiles(files);
}

function handleFileSelect(e) {
    const files = e.target.files;
    handleFiles(files);
}

// File Processing
function handleFiles(files) {
    if (!files.length || isProcessing) return;

    const file = files[0];
    if (!isValidImageFile(file)) {
        showError('Please select a valid image file (JPEG, PNG, WebP).');
        return;
    }

    if (file.size > 10 * 1024 * 1024) {
        showError('File size too large. Maximum 10MB allowed.');
        return;
    }

    displayImagePreview(file);
    window.uploadedFile = file;

    // Show success message
    showSuccessMessage('Image uploaded successfully! Starting analysis...');

    // Auto-predict after preview
    setTimeout(() => {
        if (!isProcessing) {
            predictImage();
        }
    }, 1500);
}

function isValidImageFile(file) {
    const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp'];
    return validTypes.includes(file.type.toLowerCase());
}

function displayImagePreview(file) {
    const reader = new FileReader();
    reader.onload = function(e) {
        if (previewImg && previewSection) {
            previewImg.src = e.target.result;
            previewImg.style.opacity = '0';
            previewSection.classList.remove('hidden');
            setTimeout(() => {
                previewImg.style.transition = 'opacity 0.5s ease';
                previewImg.style.opacity = '1';
                // Scroll to preview section
                previewSection.scrollIntoView({ 
                    behavior: 'smooth', 
                    block: 'center' 
                });
            }, 100);

            // Hide other sections
            if (resultSection) resultSection.classList.add('hidden');
            if (processingSection) processingSection.classList.add('hidden');
            if (loading) loading.classList.add('hidden');
        }
    };

    reader.onerror = function() {
        showError('Failed to read image file.');
    };

    reader.readAsDataURL(file);
}

// API Communication with processing flow
async function predictImage() {
    if (!window.uploadedFile || isProcessing) {
        if (!window.uploadedFile) {
            showError('No file selected.');
        }
        return;
    }

    isProcessing = true;

    try {
        // Show processing steps
        showProcessingSteps();

        // Hide other sections
        if (resultSection) resultSection.classList.add('hidden');
        if (loading) loading.classList.add('hidden');

        // Simulate processing steps
        await simulateProcessingSteps();

        // Make API call
        const formData = new FormData();
        formData.append('image', window.uploadedFile);

        const response = await fetch('/predict', {
            method: 'POST',
            body: formData,
            headers: {
                'Accept': 'application/json',
            }
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();
        if (result.error) {
            throw new Error(result.error);
        }

        // Complete final step
        completeProcessingStep('step4');

        // Show results after brief delay
        setTimeout(() => {
            hideProcessingSteps();
            displayResults(result);
        }, 1000);

    } catch (error) {
        console.error('Prediction error:', error);
        hideProcessingSteps();
        showError(`Failed to process image: ${error.message}`);
    } finally {
        isProcessing = false;
    }
}

// Processing Steps Functions
function showProcessingSteps() {
    if (!processingSection) return;

    processingSection.classList.remove('hidden');
    processingSection.scrollIntoView({ 
        behavior: 'smooth', 
        block: 'center' 
    });

    // Reset all steps
    resetProcessingSteps();

    // Mark first step as completed
    setTimeout(() => {
        completeProcessingStep('step1');
    }, 500);
}

function hideProcessingSteps() {
    if (processingSection) {
        processingSection.classList.add('hidden');
    }
}

function resetProcessingSteps() {
    const steps = ['step1', 'step2', 'step3', 'step4'];
    steps.forEach(stepId => {
        const step = document.getElementById(stepId);
        if (step) {
            step.classList.remove('active', 'completed');
            const status = step.querySelector('.step-status');
            if (status) {
                status.textContent = '⏳';
            }
        }
    });
}

function completeProcessingStep(stepId) {
    const step = document.getElementById(stepId);
    if (step) {
        step.classList.remove('active');
        step.classList.add('completed');
        const status = step.querySelector('.step-status');
        if (status) {
            status.textContent = '✅';
        }
    }
}

function activateProcessingStep(stepId) {
    const step = document.getElementById(stepId);
    if (step) {
        step.classList.add('active');
        const status = step.querySelector('.step-status');
        if (status) {
            status.textContent = '⚙️';
        }
    }
}

async function simulateProcessingSteps() {
    return new Promise((resolve) => {
        // Step 2: Preprocessing
        setTimeout(() => {
            activateProcessingStep('step2');
        }, 800);

        setTimeout(() => {
            completeProcessingStep('step2');
            activateProcessingStep('step3');
        }, 1800);

        // Step 3: AI Analysis
        setTimeout(() => {
            completeProcessingStep('step3');
            activateProcessingStep('step4');
        }, 3500);

        // Resolve after processing
        setTimeout(() => {
            resolve();
        }, 4000);
    });
}

// Results Display
function displayResults(result) {
    if (!classNameEl || !confidenceEl || !resultSection) return;

    // Animate result values
    classNameEl.style.opacity = '0';
    confidenceEl.style.opacity = '0';

    setTimeout(() => {
        classNameEl.textContent = result.className || 'Unknown';
        confidenceEl.textContent = result.confidence ? result.confidence.toFixed(2) : '0.00';
        classNameEl.style.transition = 'opacity 0.5s ease';
        confidenceEl.style.transition = 'opacity 0.5s ease';
        classNameEl.style.opacity = '1';
        confidenceEl.style.opacity = '1';
    }, 200);

    // Render chart if available
    if (result.predictions && Array.isArray(result.predictions)) {
        setTimeout(() => {
            renderChart(result.predictions);
        }, 500);
    }

    // Show results section
    setTimeout(() => {
        resultSection.classList.remove('hidden');
        resultSection.style.opacity = '0';
        resultSection.style.transform = 'translateY(20px)';

        // Scroll to results
        resultSection.scrollIntoView({ 
            behavior: 'smooth', 
            block: 'center' 
        });

        setTimeout(() => {
            resultSection.style.transition = 'all 0.8s ease';
            resultSection.style.opacity = '1';
            resultSection.style.transform = 'translateY(0)';
        }, 100);

        // Add celebration for high confidence
        if (result.confidence && result.confidence > 0.8) {
            setTimeout(() => {
                addCelebrationEffect();
            }, 800);
        }
    }, 600);
}

function addCelebrationEffect() {
    const celebration = document.createElement('div');
    celebration.innerHTML = '🎉';
    celebration.style.cssText = `
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        font-size: 4rem;
        z-index: 1002;
        pointer-events: none;
        animation: celebrate 2s ease-out forwards;
    `;

    // Add animation styles
    const style = document.createElement('style');
    style.textContent = `
        @keyframes celebrate {
            0% { transform: translate(-50%, -50%) scale(0) rotate(0deg); opacity: 0; }
            50% { transform: translate(-50%, -50%) scale(1.2) rotate(180deg); opacity: 1; }
            100% { transform: translate(-50%, -50%) scale(0) rotate(360deg); opacity: 0; }
        }
    `;

    if (!document.head.querySelector('style[data-celebration]')) {
        style.setAttribute('data-celebration', 'true');
        document.head.appendChild(style);
    }

    document.body.appendChild(celebration);

    setTimeout(() => {
        if (document.body.contains(celebration)) {
            document.body.removeChild(celebration);
        }
    }, 2000);
}

// Chart Rendering
function renderChart(predictions) {
    if (!chartCanvas) return;

    const ctx = chartCanvas.getContext('2d');
    if (currentChart) {
        currentChart.destroy();
        currentChart = null;
    }

    const classNames = [
        "Adas", "Andaliman", "Asam Jawa", "Bawang Bombai", "Bawang Merah", "Bawang Putih",
        "Biji Ketumbar", "Bukan Rempah", "Bunga Lawang", "Cengkeh", "Daun Jeruk", "Daun Kemangi",
        "Daun Ketumbar", "Daun Salam", "Jahe", "Jinten", "Kapulaga", "Kayu Manis", "Kayu Secang",
        "Kemiri", "Kemukus", "Kencur", "Kluwek", "Kunyit", "Lada", "Lengkuas", "Pala", "Saffron",
        "Serai", "Vanili", "Wijen"
    ];

    const topIndices = predictions
        .map((prob, index) => ({ prob, index }))
        .sort((a, b) => b.prob - a.prob)
        .slice(0, 8);

    const labels = topIndices.map(item => classNames[item.index] || `Class ${item.index}`);
    const data = topIndices.map(item => (item.prob * 100).toFixed(2));
    const colors = generateGradientColors(data.length);

    currentChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Confidence (%)',
                data: data,
                backgroundColor: colors.background,
                borderColor: colors.border,
                borderWidth: 2,
                borderRadius: 8,
                borderSkipped: false,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: 'rgba(15, 15, 35, 0.9)',
                    titleColor: '#a855f7',
                    bodyColor: '#e0e0e0',
                    borderColor: '#a855f7',
                    borderWidth: 1,
                    cornerRadius: 8,
                    callbacks: {
                        label: function(context) {
                            return `Confidence: ${context.parsed.y}%`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    ticks: {
                        color: '#c7d2fe',
                        font: { size: 11, weight: '500' },
                        maxRotation: 45,
                        minRotation: 45
                    },
                    grid: {
                        color: 'rgba(168, 85, 247, 0.1)',
                        drawBorder: false
                    }
                },
                y: {
                    beginAtZero: true,
                    max: 100,
                    ticks: {
                        color: '#c7d2fe',
                        font: { size: 11 },
                        callback: function(value) {
                            return value + '%';
                        }
                    },
                    grid: {
                        color: 'rgba(168, 85, 247, 0.1)',
                        drawBorder: false
                    },
                    title: {
                        display: true,
                        text: 'Confidence Level (%)',
                        color: '#a855f7',
                        font: { size: 12, weight: '600' }
                    }
                }
            },
            animation: {
                duration: 1500,
                easing: 'easeInOutCubic'
            }
        }
    });
}

function generateGradientColors(count) {
    const background = [];
    const border = [];

    for (let i = 0; i < count; i++) {
        const hue = (260 + i * 20) % 360;
        const saturation = 70 + (i * 5);
        const lightness = 60 + (i * 3);
        background.push(`hsla(${hue}, ${saturation}%, ${lightness}%, 0.7)`);
        border.push(`hsla(${hue}, ${saturation}%, ${lightness - 10}%, 1)`);
    }

    return { background, border };
}

// Notification Functions
function showError(message) {
    showNotification(message, 'error');
}

function showSuccessMessage(message) {
    showNotification(message, 'success');
}

function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    const bgColor = type === 'error' ? 'linear-gradient(135deg, #ef4444, #dc2626)' : 
                   type === 'success' ? 'linear-gradient(135deg, #22c55e, #16a34a)' :
                   'linear-gradient(135deg, #3b82f6, #2563eb)';

    notification.style.cssText = `
        position: fixed;
        top: 100px;
        right: 20px;
        background: ${bgColor};
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        z-index: 1001;
        font-weight: 600;
        max-width: 400px;
        opacity: 0;
        transform: translateX(100%);
        transition: all 0.3s ease;
        font-family: inherit;
    `;

    notification.textContent = message;
    document.body.appendChild(notification);

    setTimeout(() => {
        notification.style.opacity = '1';
        notification.style.transform = 'translateX(0)';
    }, 100);

    const duration = type === 'error' ? 5000 : 3000;
    setTimeout(() => {
        notification.style.opacity = '0';
        notification.style.transform = 'translateX(100%)';
        setTimeout(() => {
            if (document.body.contains(notification)) {
                document.body.removeChild(notification);
            }
        }, 300);
    }, duration);
}

// Additional Effects
function addScrollEffects() {
    window.addEventListener('scroll', () => {
        const scrolled = window.pageYOffset;
        const header = document.querySelector('.header');

        // Header background effect
        if (header) {
            if (scrolled > 50) {
                header.style.background = 'rgba(15, 15, 35, 0.95)';
                header.style.backdropFilter = 'blur(20px)';
            } else {
                header.style.background = 'rgba(15, 15, 35, 0.9)';
                header.style.backdropFilter = 'blur(10px)';
            }
        }
    });

    // Intersection Observer for animations
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, observerOptions);

    const sections = document.querySelectorAll('section');
    sections.forEach(section => {
        section.style.opacity = '0';
        section.style.transform = 'translateY(30px)';
        section.style.transition = 'all 0.8s ease';
        observer.observe(section);
    });
}

function addScrollToTopButton() {
    const scrollBtn = document.createElement('button');
    scrollBtn.innerHTML = '↑';
    scrollBtn.style.cssText = `
        position: fixed;
        bottom: 30px;
        right: 30px;
        width: 50px;
        height: 50px;
        border-radius: 50%;
        background: linear-gradient(135deg, #7c3aed, #a855f7);
        color: white;
        border: none;
        font-size: 1.5rem;
        font-weight: bold;
        cursor: pointer;
        z-index: 1000;
        opacity: 0;
        transform: translateY(100px);
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(168, 85, 247, 0.3);
    `;

    document.body.appendChild(scrollBtn);

    scrollBtn.addEventListener('click', () => {
        window.scrollTo({ top: 0, behavior: 'smooth' });
    });

    window.addEventListener('scroll', () => {
        if (window.scrollY > 300) {
            scrollBtn.style.opacity = '1';
            scrollBtn.style.transform = 'translateY(0)';
        } else {
            scrollBtn.style.opacity = '0';
            scrollBtn.style.transform = 'translateY(100px)';
        }
    });
}

function addLoadingEffects() {
    // Pulse effect for browse button when no file uploaded
    setInterval(() => {
        if (!window.uploadedFile && browseBtn && !isProcessing) {
            browseBtn.style.transition = 'transform 0.2s ease';
            browseBtn.style.transform = 'scale(1.05)';
            setTimeout(() => {
                browseBtn.style.transform = 'scale(1)';
            }, 200);
        }
    }, 3000);
}

// Setup all event listeners
function setupEventListeners() {
    // Upload functionality
    if (browseBtn) {
        browseBtn.addEventListener('click', () => {
            fileInput.click();
        });
    }

    if (fileInput) {
        fileInput.addEventListener('change', handleFileSelect);
    }

    if (dropZone) {
        // Drag and drop events
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, preventDefaults, false);
        });

        ['dragenter', 'dragover'].forEach(eventName => {
            dropZone.addEventListener(eventName, highlight, false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, unhighlight, false);
        });

        dropZone.addEventListener('drop', handleDrop, false);
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    init();

    // Additional initialization
    console.log('Spice Detection App initialized successfully');

    // Check for browser compatibility
    if (!window.FileReader) {
        showError('Your browser does not support file upload. Please use a modern browser.');
    }

    if (!window.fetch) {
        showError('Your browser does not support fetch API. Please use a modern browser.');
    }
});

// Error boundary for unhandled errors
window.addEventListener('error', (event) => {
    console.error('Global error:', event.error);
    handleError(event.error, 'Global');
});

window.addEventListener('unhandledrejection', (event) => {
    console.error('Unhandled promise rejection:', event.reason);
    handleError(event.reason, 'Promise');
    event.preventDefault();
});

// Handle window resize
window.addEventListener('resize', () => {
    if (currentChart) {
        currentChart.resize();
    }
});

// Keyboard accessibility
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && navMenu && navMenu.classList.contains('active')) {
        if (hamburger) hamburger.classList.remove('active');
        navMenu.classList.remove('active');
    }

    if (e.key === 'Enter' && e.target === browseBtn && fileInput) {
        fileInput.click();
    }
});

// Handle paste for image upload
document.addEventListener('paste', (e) => {
    const items = e.clipboardData?.items;
    if (!items || isProcessing) return;

    for (let item of items) {
        if (item.type.indexOf('image') !== -1) {
            const file = item.getAsFile();
            if (file) {
                handleFiles([file]);
            }
            break;
        }
    }
});