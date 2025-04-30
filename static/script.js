// index page js

const form = document.getElementById('interest-form');

form.addEventListener('submit', async function(event) {
    event.preventDefault();

    const selectedInterests = Array.from(document.querySelectorAll('input[name="interest"]:checked'))
                                   .map(cb => cb.value);

    const response = await fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ interests: selectedInterests })
    });

    const data = await response.json();
    alert(`Suggested Career: ${data.predicted_career}`);
});
