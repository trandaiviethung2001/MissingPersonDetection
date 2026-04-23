const peopleList = document.getElementById("people-list");
const trainingForm = document.getElementById("training-form");
const trainButton = document.getElementById("train-button");
const resultBox = document.getElementById("training-result");

async function fetchPeople() {
    peopleList.innerHTML = "<p class='card-note'>Loading people...</p>";
    const response = await fetch("/api/training/people");
    const data = await response.json();

    if (!data.people.length) {
        peopleList.innerHTML = "<p class='card-note'>No people added yet.</p>";
        return;
    }

    peopleList.innerHTML = data.people.map((person) => `
        <div class="person-item">
            <div class="person-item-title">${person.person_id} - ${person.name}</div>
            <div class="person-item-meta">${person.image_count} image(s)</div>
        </div>
    `).join("");
}

trainingForm.addEventListener("submit", async (event) => {
    event.preventDefault();

    const name = document.getElementById("person-name").value.trim();
    const files = document.getElementById("person-images").files;
    if (!name || !files.length) {
        resultBox.textContent = "Name and photos are required.";
        return;
    }

    const formData = new FormData();
    formData.append("name", name);
    for (const file of files) {
        formData.append("images", file);
    }

    resultBox.textContent = "Uploading person data...";
    const response = await fetch("/api/training/person", {
        method: "POST",
        body: formData
    });
    const data = await response.json();

    if (!response.ok || !data.ok) {
        resultBox.textContent = data.detail || "Failed to add person.";
        return;
    }

    resultBox.textContent = `Added ${data.person.name} as ${data.person.person_id}.`;
    trainingForm.reset();
    fetchPeople();
});

trainButton.addEventListener("click", async () => {
    resultBox.textContent = "Building embeddings.pkl...";
    const response = await fetch("/api/training/run", {
        method: "POST"
    });
    const data = await response.json();

    if (!response.ok || !data.ok) {
        resultBox.textContent = data.detail || "Training failed.";
        return;
    }

    resultBox.textContent = `Done. ${data.training.people_count} person(s), ${data.training.embedding_count} embedding(s). Latest file: ${data.training.latest_output}`;
});

fetchPeople();
