let currentCaptions = [];

document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('uploadForm');
    const fileInput = document.getElementById('imageInput');
    const suggestionInput = document.getElementById('suggestionInput');
    const captionsContainer = document.getElementById('captionsContainer');
    const imagePreview = document.getElementById('imagePreview');
    const loadingIndicator = document.getElementById('loadingIndicator');
    const errorMessage = document.getElementById('errorMessage');

    // Handle file input change to show preview
    fileInput.addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                imagePreview.src = e.target.result;
                imagePreview.style.display = 'block';
                captionsContainer.innerHTML = '';
                errorMessage.textContent = '';
            };
            reader.readAsDataURL(file);
        }
    });

    // Handle form submission
    uploadForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const file = fileInput.files[0];
        if (!file) {
            errorMessage.textContent = 'Please select an image file';
            return;
        }

        // Show loading indicator
        loadingIndicator.style.display = 'block';
        errorMessage.textContent = '';

        const formData = new FormData();
        formData.append('file', file);
        formData.append('suggestion', suggestionInput.value);

        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'Failed to generate captions');
            }

            // Update image preview with the processed image
            if (data.image_data) {
                imagePreview.src = `data:image/jpeg;base64,${data.image_data}`;
                imagePreview.style.display = 'block';
            }

            // Display captions
            captionsContainer.innerHTML = '';
            data.captions.forEach(caption => {
                const captionElement = document.createElement('div');
                captionElement.className = 'caption-item';
                captionElement.innerHTML = `
                    <p class="caption-text">${caption.text}</p>
                    <button class="select-caption" data-id="${caption.id}">Select</button>
                `;
                captionsContainer.appendChild(captionElement);
            });

        } catch (error) {
            errorMessage.textContent = error.message;
            console.error('Error:', error);
        } finally {
            loadingIndicator.style.display = 'none';
        }
    });

    // Handle caption selection
    captionsContainer.addEventListener('click', async function(e) {
        if (e.target.classList.contains('select-caption')) {
            const captionId = e.target.dataset.id;
            try {
                const response = await fetch('/select-caption', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ caption_id: captionId })
                });

                if (!response.ok) {
                    throw new Error('Failed to select caption');
                }

                // Update UI to show selected caption
                const captionItems = document.querySelectorAll('.caption-item');
                captionItems.forEach(item => {
                    item.classList.remove('selected');
                });
                e.target.closest('.caption-item').classList.add('selected');
            } catch (error) {
                errorMessage.textContent = 'Failed to select caption';
                console.error('Error:', error);
            }
        }
    });
}); 