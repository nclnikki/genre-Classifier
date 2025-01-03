const aboutLink = document.getElementById("aboutLink");
const predictLink = document.getElementById("predictLink");
const aboutPage = document.getElementById("aboutPage");
const predictPage = document.getElementById("predictPage");
const fileInput = document.getElementById("fileInput");
const fileName = document.getElementById("fileName");
const fileSize = document.getElementById("fileSize");
const removeFile = document.getElementById("removeFile");
const playButton = document.getElementById("playButton");
const audioPlayer = document.getElementById("audioPlayer");
const audioSource = document.getElementById("audioSource");
const resultDiv = document.getElementById("result");

aboutLink.addEventListener("click", () => {
  aboutPage.classList.add("active");
  predictPage.classList.remove("active");
});

predictLink.addEventListener("click", () => {
  predictPage.classList.add("active");
  aboutPage.classList.remove("active");
});

fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) {
    const file = fileInput.files[0];
    fileName.textContent = file.name;
    fileSize.textContent = (file.size / 1024 / 1024).toFixed(1) + "MB";
    audioSource.src = URL.createObjectURL(file);
    audioPlayer.load();
    audioPlayer.hidden = false;
  }
});

removeFile.addEventListener("click", () => {
  fileInput.value = "";
  fileName.textContent = "No file selected";
  fileSize.textContent = "";
  audioSource.src = "";
  audioPlayer.hidden = true;
});


document.getElementById("analyzeButton").addEventListener("click", () => {
    const fileInput = document.getElementById("fileInput");
    const resultDiv = document.getElementById("result");
    const loadingDiv = document.getElementById("loading");
  
    if (!fileInput.files[0]) {
      resultDiv.textContent = "Please upload a file!";
      return;
    }
  
    // Show the loading animation
    loadingDiv.style.display = "block";
    resultDiv.textContent = "";
  
    const formData = new FormData();
    formData.append("file", fileInput.files[0]);
  
    fetch("http://127.0.0.1:5000/predict", {
      method: "POST",
      body: formData,
    })
      .then((response) => response.json())
      .then((data) => {
        // Hide the loading animation
        loadingDiv.style.display = "none";
  
        if (data.error) {
          resultDiv.textContent = "Error: " + data.error;
        } else {
          resultDiv.textContent = "Predicted Genre: " + data.genre;
        }
      })
      .catch((error) => {
        // Hide the loading animation
        loadingDiv.style.display = "none";
        resultDiv.textContent = "Error: " + error.message;
      });
  });
  