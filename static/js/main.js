// Main JavaScript for CSV Route Optimization UI

// Global variables
let currentOptimizationJob = null;
let progressInterval = null;

// Utility functions
function showAlert(message, type = 'info') {
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
    alertDiv.innerHTML = `
        <i class="fas fa-${getAlertIcon(type)} me-2"></i>
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    const container = document.querySelector('.container');
    container.insertBefore(alertDiv, container.firstChild);
    
    // Auto-dismiss after 5 seconds
    setTimeout(() => {
        if (alertDiv.parentNode) {
            alertDiv.remove();
        }
    }, 5000);
}

function getAlertIcon(type) {
    const icons = {
        'success': 'check-circle',
        'danger': 'exclamation-triangle',
        'warning': 'exclamation-circle',
        'info': 'info-circle'
    };
    return icons[type] || 'info-circle';
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function formatDateTime(dateString) {
    const date = new Date(dateString);
    return date.toLocaleString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    });
}

// File upload handling
function initializeFileUpload() {
    const fileInput = document.getElementById('file');
    const uploadArea = document.querySelector('.upload-area');
    
    if (!fileInput || !uploadArea) return;
    
    // Drag and drop functionality
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });
    
    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });
    
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            fileInput.files = files;
            handleFileSelect(files[0]);
        }
    });
    
    // File input change
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });
    
    // Click to upload
    uploadArea.addEventListener('click', () => {
        fileInput.click();
    });
}

function handleFileSelect(file) {
    // Validate file type
    if (!file.name.toLowerCase().endsWith('.csv')) {
        showAlert('Please select a CSV file.', 'danger');
        return;
    }
    
    // Validate file size (10MB limit)
    if (file.size > 10 * 1024 * 1024) {
        showAlert('File size must be less than 10MB.', 'danger');
        return;
    }
    
    // Show file info
    const fileInfo = document.getElementById('file-info');
    if (fileInfo) {
        fileInfo.innerHTML = `
            <div class="d-flex align-items-center">
                <i class="fas fa-file-csv text-success me-2"></i>
                <div>
                    <div class="fw-bold">${file.name}</div>
                    <small class="text-muted">${formatFileSize(file.size)}</small>
                </div>
            </div>
        `;
    }
    
    // Enable upload button
    const uploadBtn = document.getElementById('upload-btn');
    if (uploadBtn) {
        uploadBtn.disabled = false;
    }
}

// Vehicle management
function initializeVehicleManagement() {
    // Add vehicle form
    const addForm = document.getElementById('add-vehicle-form');
    if (addForm) {
        addForm.addEventListener('submit', handleAddVehicle);
    }
    
    // Edit vehicle form
    const editForm = document.getElementById('edit-vehicle-form');
    if (editForm) {
        editForm.addEventListener('submit', handleEditVehicle);
    }
}

function handleAddVehicle(e) {
    e.preventDefault();
    
    const formData = new FormData(e.target);
    const vehicleData = {
        vehicle_type: formData.get('vehicle_type'),
        capacity_kg: parseFloat(formData.get('capacity_kg')),
        max_locations: parseInt(formData.get('max_locations')),
        cost_per_km: parseFloat(formData.get('cost_per_km')),
        max_distance_km: parseFloat(formData.get('max_distance_km'))
    };
    
    // Validate data
    if (!validateVehicleData(vehicleData)) {
        return;
    }
    
    // Submit to API
    submitVehicleData('POST', vehicleData)
        .then(() => {
            showAlert('Vehicle added successfully!', 'success');
            bootstrap.Modal.getInstance(document.getElementById('addVehicleModal')).hide();
            e.target.reset();
            // Refresh vehicle list
            setTimeout(() => location.reload(), 1000);
        })
        .catch(error => {
            showAlert('Error adding vehicle: ' + error.message, 'danger');
        });
}

function handleEditVehicle(e) {
    e.preventDefault();
    
    const formData = new FormData(e.target);
    const vehicleData = {
        id: parseInt(formData.get('vehicle_id')),
        vehicle_type: formData.get('vehicle_type'),
        capacity_kg: parseFloat(formData.get('capacity_kg')),
        max_locations: parseInt(formData.get('max_locations')),
        cost_per_km: parseFloat(formData.get('cost_per_km')),
        max_distance_km: parseFloat(formData.get('max_distance_km'))
    };
    
    // Validate data
    if (!validateVehicleData(vehicleData)) {
        return;
    }
    
    // Submit to API
    submitVehicleData('PUT', vehicleData)
        .then(() => {
            showAlert('Vehicle updated successfully!', 'success');
            bootstrap.Modal.getInstance(document.getElementById('editVehicleModal')).hide();
            // Refresh vehicle list
            setTimeout(() => location.reload(), 1000);
        })
        .catch(error => {
            showAlert('Error updating vehicle: ' + error.message, 'danger');
        });
}

function validateVehicleData(data) {
    if (!data.vehicle_type || data.vehicle_type.trim() === '') {
        showAlert('Vehicle type is required.', 'danger');
        return false;
    }
    
    if (!data.capacity_kg || data.capacity_kg <= 0) {
        showAlert('Capacity must be greater than 0.', 'danger');
        return false;
    }
    
    if (!data.max_locations || data.max_locations <= 0) {
        showAlert('Max locations must be greater than 0.', 'danger');
        return false;
    }
    
    if (!data.cost_per_km || data.cost_per_km <= 0) {
        showAlert('Cost per km must be greater than 0.', 'danger');
        return false;
    }
    
    if (!data.max_distance_km || data.max_distance_km <= 0) {
        showAlert('Max distance must be greater than 0.', 'danger');
        return false;
    }
    
    return true;
}

function submitVehicleData(method, data) {
    const hubId = document.querySelector('[data-hub-id]')?.dataset.hubId || 
                  window.location.pathname.match(/\/vehicle-config\/.*\/(\d+)/)?.[1];
    
    if (!hubId) {
        return Promise.reject(new Error('Hub ID not found'));
    }
    
    const url = method === 'POST' ? `/api/vehicles/${hubId}` : `/api/vehicles/${hubId}`;
    
    return fetch(url, {
        method: method,
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data)
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        return response.json();
    })
    .then(result => {
        if (result.status !== 'success') {
            throw new Error(result.message || 'Unknown error');
        }
        return result;
    });
}

// Optimization tracking
function startOptimizationTracking(jobId) {
    currentOptimizationJob = jobId;
    
    progressInterval = setInterval(() => {
        fetch(`/api/optimization-status/${jobId}`)
            .then(response => response.json())
            .then(data => {
                updateOptimizationProgress(data);
                
                if (data.status === 'COMPLETED' || data.status === 'ERROR') {
                    stopOptimizationTracking();
                }
            })
            .catch(error => {
                console.error('Error tracking optimization:', error);
                showAlert('Error tracking optimization progress.', 'danger');
                stopOptimizationTracking();
            });
    }, 2000);
}

function updateOptimizationProgress(data) {
    const statusText = document.getElementById('optimization-status');
    const detailsText = document.getElementById('optimization-details');
    
    if (statusText) {
        switch (data.status) {
            case 'PENDING':
                statusText.textContent = 'Queued for processing...';
                break;
            case 'RUNNING':
                statusText.textContent = 'Optimizing routes...';
                break;
            case 'COMPLETED':
                statusText.textContent = 'Optimization Complete!';
                break;
            case 'ERROR':
                statusText.textContent = 'Optimization Failed';
                break;
            default:
                statusText.textContent = 'Unknown status';
        }
    }
    
    if (detailsText) {
        if (data.status === 'ERROR') {
            detailsText.textContent = data.error_message || 'An error occurred during optimization.';
        } else if (data.status === 'COMPLETED') {
            detailsText.textContent = 'Your routes have been optimized successfully.';
        } else if (data.status === 'RUNNING' && data.status_message) {
            detailsText.textContent = data.status_message;
        } else {
            detailsText.textContent = getOptimizationDetails(data.status);
        }
    }
    
    // Update buttons
    const cancelBtn = document.getElementById('cancel-btn');
    const downloadBtn = document.getElementById('download-btn');
    
    if (data.status === 'COMPLETED') {
        if (cancelBtn) cancelBtn.style.display = 'none';
        if (downloadBtn) downloadBtn.style.display = 'inline-block';
    } else if (data.status === 'ERROR') {
        if (cancelBtn) {
            cancelBtn.textContent = 'Close';
            cancelBtn.onclick = () => bootstrap.Modal.getInstance(document.getElementById('optimizationModal')).hide();
        }
    }
}

function getOptimizationDetails(status) {
    const details = {
        'PENDING': 'Your optimization request is queued and will start shortly.',
        'RUNNING': 'Processing pickup locations and calculating optimal routes...',
        'COMPLETED': 'All routes have been optimized successfully.',
        'ERROR': 'An error occurred during optimization.'
    };
    
    return details[status] || 'Processing...';
}

function stopOptimizationTracking() {
    if (progressInterval) {
        clearInterval(progressInterval);
        progressInterval = null;
    }
    currentOptimizationJob = null;
}

// Dashboard data loading
function loadDashboardData() {
    // This would typically fetch data from API endpoints
    // For now, we'll simulate data loading
    
    const stats = {
        totalJobs: 12,
        completedJobs: 8,
        runningJobs: 2,
        failedJobs: 2
    };
    
    // Update stats
    document.getElementById('total-jobs').textContent = stats.totalJobs;
    document.getElementById('completed-jobs').textContent = stats.completedJobs;
    document.getElementById('running-jobs').textContent = stats.runningJobs;
    document.getElementById('failed-jobs').textContent = stats.failedJobs;
    
    // Update jobs table
    updateJobsTable();
}

function updateJobsTable() {
    const tbody = document.querySelector('#jobs-table tbody');
    if (!tbody) return;
    
    // Sample data - in real implementation, this would come from API
    const jobs = [
        {
            id: 'job-001',
            hubId: 8,
            status: 'COMPLETED',
            status_message: 'Route optimization completed successfully',
            created: '2024-01-15T10:30:00',
            hasResults: true
        },
        {
            id: 'job-002',
            hubId: 8,
            status: 'RUNNING',
            status_message: 'Running HDBSCAN clustering algorithm...',
            created: '2024-01-15T11:15:00',
            hasResults: false
        }
    ];
    
    tbody.innerHTML = jobs.map(job => `
        <tr>
            <td><code>${job.id}</code></td>
            <td>${job.hubId}</td>
            <td>
                <span class="badge bg-${getStatusColor(job.status)}">
                    <span class="status-indicator ${job.status.toLowerCase()}"></span>
                    ${job.status}
                </span>
            </td>
            <td>
                <div class="status-message ${job.status.toLowerCase()}">
                    ${job.status_message || getOptimizationDetails(job.status)}
                </div>
            </td>
            <td>${formatDateTime(job.created)}</td>
            <td>
                ${job.hasResults ? 
                    `<a href="/download/${job.id}" class="btn btn-sm btn-outline-primary">
                        <i class="fas fa-download"></i> Download
                    </a>` :
                    `<button class="btn btn-sm btn-outline-secondary" disabled>
                        <i class="fas fa-spinner fa-spin"></i> Running
                    </button>`
                }
            </td>
        </tr>
    `).join('');
}

function getStatusColor(status) {
    const colors = {
        'PENDING': 'warning',
        'RUNNING': 'info',
        'COMPLETED': 'success',
        'ERROR': 'danger'
    };
    return colors[status] || 'secondary';
}

// Initialize everything when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Initialize components based on current page
    if (document.getElementById('file')) {
        initializeFileUpload();
    }
    
    if (document.getElementById('add-vehicle-form')) {
        initializeVehicleManagement();
    }
    
    if (document.getElementById('total-jobs')) {
        loadDashboardData();
        // Refresh dashboard data every 30 seconds
        setInterval(loadDashboardData, 30000);
    }
    
    // Initialize tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    // Initialize popovers
    const popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    popoverTriggerList.map(function (popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });
});

// Cleanup on page unload
window.addEventListener('beforeunload', function() {
    stopOptimizationTracking();
});
