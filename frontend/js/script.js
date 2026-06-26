// Indonesian Spice Detection System - Modern JavaScript

document.addEventListener('DOMContentLoaded', () => {
  initNavigation();
  initUpload();
  initSmoothScroll();
});

// Navigation
function initNavigation() {
  const hamburger = document.querySelector('.hamburger');
  const navMenu = document.querySelector('.nav-menu');
  const navLinks = document.querySelectorAll('.nav-link');

  if (hamburger && navMenu) {
    hamburger.addEventListener('click', () => {
      hamburger.classList.toggle('active');
      navMenu.classList.toggle('active');
    });
  }

  navLinks.forEach(link => {
    link.addEventListener('click', (e) => {
      e.preventDefault();
      hamburger?.classList.remove('active');
      navMenu?.classList.remove('active');
      
      const targetId = link.getAttribute('href');
      const targetSection = document.querySelector(targetId);
      if (targetSection) {
        const header = document.querySelector('.header');
        const headerHeight = header ? header.offsetHeight : 80;
        const targetPosition = targetSection.offsetTop - headerHeight - 20;
        window.scrollTo({ top: targetPosition, behavior: 'smooth' });
      }
    });
  });

  window.addEventListener('scroll', updateActiveNavItem);
}

function updateActiveNavItem() {
  const sections = document.querySelectorAll('section[id]');
  const navLinks = document.querySelectorAll('.nav-link');
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
    link.classList.toggle('active', link.getAttribute('href') === `#${currentSection}`);
  });
}

// Upload functionality
function initUpload() {
  const dropZone = document.getElementById('dropZone');
  const fileInput = document.getElementById('fileInput');
  const browseBtn = document.getElementById('browseBtn');

  if (!dropZone || !fileInput || !browseBtn) return;

  ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, preventDefaults, false);
  });

  function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
  }

  ['dragenter', 'dragover'].forEach(eventName => {
    dropZone.addEventListener(eventName, () => dropZone.classList.add('highlight'), false);
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, () => dropZone.classList.remove('highlight'), false);
  });

  dropZone.addEventListener('drop', handleDrop, false);
  browseBtn.addEventListener('click', () => fileInput.click());
  fileInput.addEventListener('change', handleFiles);

  function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;
    handleFiles({ target: { files } });
  }

  function handleFiles(e) {
    const files = e.target.files;
    if (files.length > 0) {
      uploadFile(files[0]);
    }
  }
}

function uploadFile(file) {
  const previewSection = document.getElementById('previewSection');
  const processingSection = document.getElementById('processingSection');
  const resultSection = document.getElementById('resultSection');
  const loading = document.getElementById('loading');
  const previewImg = document.getElementById('previewImg');

  // Reset sections
  previewSection?.classList.add('hidden');
  processingSection?.classList.add('hidden');
  resultSection?.classList.add('hidden');
  loading?.classList.remove('hidden');

  // Show preview
  const reader = new FileReader();
  reader.onload = (e) => {
    previewImg.src = e.target.result;
    previewSection?.classList.remove('hidden');
  };
  reader.readAsDataURL(file);

  // Simulate processing steps
  setTimeout(() => updateStep(1, 'active'), 500);
  setTimeout(() => updateStep(1, 'completed'), 1500);
  setTimeout(() => updateStep(2, 'active'), 1500);
  setTimeout(() => updateStep(2, 'completed'), 2500);
  setTimeout(() => updateStep(3, 'active'), 2500);

  // Send to API
  const formData = new FormData();
  formData.append('image', file);

  fetch('/api/v1/predict', {
    method: 'POST',
    body: formData
  })
  .then(response => response.json())
  .then(data => {
    updateStep(3, 'completed');
    updateStep(4, 'completed');
    loading?.classList.add('hidden');
    processingSection?.classList.remove('hidden');
    
    displayResults(data);
  })
  .catch(error => {
    console.error('Error:', error);
    loading?.classList.add('hidden');
    alert('Error processing image. Please try again.');
  });
}

function updateStep(stepNumber, status) {
  const step = document.getElementById(`step${stepNumber}`);
  if (!step) return;
  
  step.classList.remove('active', 'completed');
  step.classList.add(status);
}

function displayResults(data) {
  const resultSection = document.getElementById('resultSection');
  const classNameEl = document.getElementById('className');
  const confidenceEl = document.getElementById('confidence');

  if (classNameEl) classNameEl.textContent = data.className || 'Unknown';
  if (confidenceEl) confidenceEl.textContent = (data.confidence || 0).toFixed(1);

  resultSection?.classList.remove('hidden');

  // Draw chart if Chart.js is available
  if (typeof Chart !== 'undefined' && data.topPredictions) {
    drawChart(data.topPredictions);
  }
}

function drawChart(predictions) {
  const ctx = document.getElementById('chart');
  if (!ctx) return;

  const existingChart = Chart.getChart(ctx);
  if (existingChart) existingChart.destroy();

  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: predictions.map(p => p.class),
      datasets: [{
        label: 'Confidence (%)',
        data: predictions.map(p => p.confidence),
        backgroundColor: 'rgba(139, 92, 246, 0.7)',
        borderColor: 'rgba(139, 92, 246, 1)',
        borderWidth: 1
      }]
    },
    options: {
      responsive: true,
      scales: {
        y: {
          beginAtZero: true,
          max: 100,
          grid: { color: 'rgba(71, 85, 105, 0.3)' },
          ticks: { color: '#cbd5e1' }
        },
        x: {
          grid: { display: false },
          ticks: { color: '#cbd5e1' }
        }
      },
      plugins: {
        legend: { display: false }
      }
    }
  });
}

// Smooth scroll for anchor links
function initSmoothScroll() {
  document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function(e) {
      const href = this.getAttribute('href');
      if (href !== '#') {
        e.preventDefault();
        const target = document.querySelector(href);
        if (target) {
          const header = document.querySelector('.header');
          const headerHeight = header ? header.offsetHeight : 80;
          const targetPosition = target.offsetTop - headerHeight - 20;
          window.scrollTo({ top: targetPosition, behavior: 'smooth' });
        }
      }
    });
  });
}
