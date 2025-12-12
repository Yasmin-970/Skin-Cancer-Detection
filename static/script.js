function previewImage() {
  var file = document.getElementById("imageInput").files[0];
  document.getElementById("result-display").innerText = "";

  if (file) {
    var reader = new FileReader();
    reader.onload = function (e) {
      var img = document.getElementById("preview-image");
      img.src = e.target.result;
      img.style.display = "block";
    };
    reader.readAsDataURL(file);
  }
}

async function analyzeImage() {
  const fileInput = document.getElementById("imageInput");
  const resultDisplay = document.getElementById("result-display");

  if (fileInput.files.length === 0) {
    alert("Please select an image first!");
    return;
  }

  const formData = new FormData();
  formData.append("file", fileInput.files[0]);

  resultDisplay.innerText = "Analyzing... Please wait";
  resultDisplay.style.color = "#555";

  try {
    const response = await fetch("http://127.0.0.1:8000/predict", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      throw new Error("Server Error");
    }

    const data = await response.json();

    const confidencePercent = (data.confidence * 100).toFixed(2);
    resultDisplay.innerText = `Diagnosis: ${data.prediction} (${confidencePercent}%)`;

    if (data.prediction === "Malignant") {
      resultDisplay.style.color = "#dc2626";
    } else {
      resultDisplay.style.color = "#10b981";
    }
  } catch (error) {
    console.error("Error:", error);
    resultDisplay.innerText =
      "Error: Could not connect to the server. Make sure Backend.py is running.";
    resultDisplay.style.color = "red";
  }
}
