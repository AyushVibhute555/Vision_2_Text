const imageInput = document.getElementById("imageInput");
const previewImage = document.getElementById("previewImage");
const result = document.getElementById("result");
const loading = document.getElementById("loading");
const button = document.getElementById("generateBtn");

imageInput.addEventListener("change", () => {
    const file = imageInput.files[0];
    if (file) {
        previewImage.src = URL.createObjectURL(file);
        previewImage.style.display = "block";
        result.innerHTML = "";
    }
});

function uploadImage() {
    const file = imageInput.files[0];
    if (!file) {
        alert("Please select an image first");
        return;
    }

    const formData = new FormData();
    formData.append("image", file);

    button.disabled = true;
    button.innerText = "🧠 Processing...";
    loading.style.display = "block";
    result.innerHTML = "";

    fetch("http://127.0.0.1:5000/generate-caption", {
        method: "POST",
        body: formData
    })
    .then(res => res.json())
    .then(data => {
        loading.style.display = "none";
        result.innerHTML = "📄 <b>Caption:</b> " + data.caption;
    })
    .catch(() => {
        loading.style.display = "none";
        alert("Error generating caption");
    })
    .finally(() => {
        button.disabled = false;
        button.innerText = "✨ Generate Caption";
    });
}
