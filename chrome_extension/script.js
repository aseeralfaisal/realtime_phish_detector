print("Phish Detector is running....")

document.addEventListener('DOMContentLoaded', () => {
    const statusElement = document.getElementById('status');

    const isPhishing = Math.random() > 0.5; 

    if (isPhishing) {
        statusElement.textContent = 'Phishing Detected';
        statusElement.classList.add('phishing');
    } else {
        statusElement.textContent = 'Safe';
        statusElement.classList.add('safe');
    }
});