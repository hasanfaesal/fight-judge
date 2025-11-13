const uploadContainer = document.getElementById("uploadContainer");
const uploadBtn = document.getElementById("uploadBtn");
const videoInput = document.getElementById("videoInput");
const videoPreview = document.getElementById("videoPreview");
const videoPlayer = document.getElementById("videoPlayer");
const fileName = document.getElementById("fileName");
const fileSize = document.getElementById("fileSize");
const analyzeBtn = document.getElementById("analyzeBtn");
const changeBtn = document.getElementById("changeBtn");

const MAX_FILE_SIZE = 500 * 1024 * 1024; // 500MB
const ALLOWED_TYPES = ["video/mp4", "video/quicktime", "video/x-msvideo"];

// Click to browse
uploadBtn.addEventListener("click", () => {
  videoInput.click();
});

uploadContainer.addEventListener("click", (e) => {
  if (e.target !== uploadBtn) {
    videoInput.click();
  }
});

// Drag and drop
uploadContainer.addEventListener("dragover", (e) => {
  e.preventDefault();
  uploadContainer.classList.remove("border-white/30");
  uploadContainer.classList.add("border-sky-500", "bg-sky-500/10");
});

uploadContainer.addEventListener("dragleave", () => {
  uploadContainer.classList.add("border-white/30");
  uploadContainer.classList.remove("border-sky-500", "bg-sky-500/10");
});

uploadContainer.addEventListener("drop", (e) => {
  e.preventDefault();
  uploadContainer.classList.add("border-white/30");
  uploadContainer.classList.remove("border-sky-500", "bg-sky-500/10");

  const files = e.dataTransfer.files;
  if (files.length > 0) {
    handleFile(files[0]);
  }
});

// File input change
videoInput.addEventListener("change", (e) => {
  if (e.target.files.length > 0) {
    handleFile(e.target.files[0]);
  }
});

// Handle file validation and preview
function handleFile(file) {
  // Validate file type (lenient - accepts any video file)
  if (
    !file.type.startsWith("video/") &&
    !file.name.match(/\.(mp4|mov|avi)$/i)
  ) {
    alert("Please upload a valid video file (MP4, MOV, or AVI)");
    return;
  }

  // Validate file size
  if (file.size > MAX_FILE_SIZE) {
    alert("File size must be less than 500MB");
    return;
  }

  // Show preview
  const url = URL.createObjectURL(file);
  videoPlayer.src = url;
  fileName.textContent = `File: ${file.name}`;
  fileSize.textContent = `Size: ${(file.size / (1024 * 1024)).toFixed(2)} MB`;

  uploadContainer.classList.add("hidden");
  videoPreview.classList.remove("hidden");
}

// Change video button
changeBtn.addEventListener("click", () => {
  videoPlayer.src = "";
  videoInput.value = "";
  uploadContainer.classList.remove("hidden");
  videoPreview.classList.add("hidden");
});

// Analyze button (placeholder)
analyzeBtn.addEventListener("click", () => {
  alert("Video analysis will be implemented here!");
  // Future: Send video to backend for processing
});
