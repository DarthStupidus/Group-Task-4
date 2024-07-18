let slider = document.getElementById("arange");
let value = document.querySelector('.value');

value.innerHTML = slider.value;

slider.addEventListener('input', function() {
    value.textContent = this.value;
});

