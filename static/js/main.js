let currentCaptions = [];

document.getElementById("upload-form").onsubmit = async function (event) {
    event.preventDefault();
    let formData = new FormData(event.target);

    // Show loading indicator
    document.getElementById("loading").style.display = "block";
    document.getElementById("descriptions").innerHTML = "";
    document.getElementById("selection-hint").style.display = "none";

    try {
        let response = await fetch("/", {
            method: "POST",
            body: formData
        });

        let data = await response.json();
        let descriptionsDiv = document.getElementById("descriptions");
        
        if (data.error) {
            descriptionsDiv.innerHTML = `<p style="color: var(--accent-color)">Error: ${data.error}</p>`;
        } else {
            currentCaptions = data.descriptions;
            descriptionsDiv.innerHTML = `<h3>Generated Descriptions:</h3>`;
            data.descriptions.forEach((caption, index) => {
                descriptionsDiv.innerHTML += `
                    <div class="description-card" data-caption-id="${caption.id}">
                        <div class="description-number">Description ${index + 1}</div>
                        <p>${caption.text}</p>
                    </div>
                `;
            });
            document.getElementById("selection-hint").style.display = "block";
        }
    } catch (error) {
        document.getElementById("descriptions").innerHTML = 
            `<p style="color: var(--accent-color)">Error: ${error.message}</p>`;
    } finally {
        // Hide loading indicator
        document.getElementById("loading").style.display = "none";
    }
};

// Handle caption selection
document.getElementById("descriptions").addEventListener("click", async function(event) {
    const card = event.target.closest(".description-card");
    if (!card) return;

    const captionId = card.dataset.captionId;
    if (!captionId) return;

    try {
        const response = await fetch("/select-caption", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ caption_id: captionId })
        });

        const data = await response.json();
        if (data.success) {
            // Update UI to show selection
            document.querySelectorAll(".description-card").forEach(c => {
                c.classList.remove("selected");
            });
            card.classList.add("selected");
            document.getElementById("selection-hint").innerHTML = "Description selected!";
        }
    } catch (error) {
        console.error("Error selecting caption:", error);
    }
});

// Handle image upload and prompt submission
document.getElementById("upload-form").onsubmit = async function (event) {
    event.preventDefault();
    let formData = new FormData(event.target);

    let response = await fetch("/", {
        method: "POST",
        body: formData
    });

    let data = await response.json();
    let captionsDiv = document.getElementById("captions");
    captionsDiv.innerHTML = "<h3>Generated Captions:</h3>";

    // Display captions
    data.captions.forEach((caption) => {
        let captionParagraph = document.createElement("p");
        captionParagraph.textContent = caption;
        captionsDiv.appendChild(captionParagraph);
    });
};

// Handle caption removal
function removeCaption(index) {
    // Remove caption from the list
    let captionsDiv = document.getElementById("captions");
    let allCaptions = captionsDiv.querySelectorAll("p");

    if (allCaptions.length > index) {
        allCaptions[index].remove();
    }

    // Also remove the caption from the dropdown list
    let select = document.getElementById("caption-select");
    select.remove(index);
}

// Add custom user caption
function addCustomCaption() {
    let userCaption = document.getElementById("user-custom-caption").value;

    if (!userCaption) {
        alert("Please enter a caption.");
        return;
    }

    // Add custom caption to the captions list
    let captionsDiv = document.getElementById("captions");
    let captionParagraph = document.createElement("p");
    captionParagraph.textContent = userCaption;

    // Create remove button for the custom caption
    let removeButton = document.createElement("span");
    removeButton.textContent = "Remove";
    removeButton.className = "remove-btn";
    removeButton.onclick = function () {
        removeCaptionFromCustom(userCaption);
    };

    // Append the remove button next to the caption
    captionParagraph.appendChild(removeButton);
    captionsDiv.appendChild(captionParagraph);

    // Add the custom caption to the dropdown list for modification
    let select = document.getElementById("caption-select");
    let option = document.createElement("option");
    option.value = userCaption;
    option.textContent = userCaption;
    select.appendChild(option);

    // Clear the input field after adding the caption
    document.getElementById("user-custom-caption").value = "";
}

// Handle custom caption removal
function removeCaptionFromCustom(caption) {
    let captionsDiv = document.getElementById("captions");
    let allCaptions = captionsDiv.querySelectorAll("p");

    allCaptions.forEach(function (captionParagraph) {
        if (captionParagraph.textContent.includes(caption)) {
            captionParagraph.remove();
        }
    });

    // Also remove the caption from the dropdown list
    let select = document.getElementById("caption-select");
    let options = select.querySelectorAll("option");

    options.forEach(function (option) {
        if (option.value === caption) {
            option.remove();
        }
    });
}

// Handle caption modification
async function modifyCaption() {
    let selectedCaption = document.getElementById("caption-select").value;
    let modificationText = document.getElementById("modification-text").value;

    // Make sure both inputs are filled
    if (!selectedCaption || !modificationText) {
        alert("Please select a caption and provide modification instructions.");
        return;
    }

    // Send request to modify caption (this could be used for logging or server-side changes)
    let response = await fetch("/modify_caption", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ selected_caption: selectedCaption, user_instruction: modificationText })
    });

    let data = await response.json();

    // Use the existing code to trigger the caption generation API again to fetch new captions
    // Reusing the existing logic for caption generation
    let formData = new FormData(document.getElementById("upload-form"));
    let generateResponse = await fetch("/", {
        method: "POST",
        body: formData
    });

    let generateData = await generateResponse.json();
    let captionsDiv = document.getElementById("captions");
    captionsDiv.innerHTML = "<h3>Generated Captions:</h3>";

    // Clear existing captions in the select dropdown
    let select = document.getElementById("caption-select");
    select.innerHTML = "";

    // Display the newly generated captions and populate the select dropdown
    generateData.captions.forEach((caption, index) => {
        let captionParagraph = document.createElement("p");
        captionParagraph.textContent = caption;

        // Create remove button for each caption
        let removeButton = document.createElement("span");
        removeButton.textContent = "Remove";
        removeButton.className = "remove-btn";
        removeButton.onclick = function () {
            removeCaption(index);
        };

        // Append the remove button next to the caption
        captionParagraph.appendChild(removeButton);
        captionsDiv.appendChild(captionParagraph);

        // Add the caption to the dropdown list for modification
        let option = document.createElement("option");
        option.value = caption;
        option.textContent = caption;
        select.appendChild(option);
    });

    // Clear the modification input fields
    document.getElementById("modification-text").value = "";
    document.getElementById("caption-select").selectedIndex = -1;

    alert("Captions successfully regenerated!");
} 