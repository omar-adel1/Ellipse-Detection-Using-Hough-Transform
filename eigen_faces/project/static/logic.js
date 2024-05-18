
// Log the values
// Get the button element
var submitBtn = document.getElementById('submit-btn');
// Get the dropdown options
var options = document.querySelectorAll('.dropdown-content a');
var div2=document.getElementById('modified');

// Variable to store the selected option
var selectedOption;

// Add an onclick event to each option
options.forEach(function(option) {
    option.onclick = function() {
        // Store the selected option
        if (this.textContent == 'DS 1'){
            selectedOption=1
        }
        else if (this.textContent == 'DS 2'){
            selectedOption=2
        }
    };
});
// Add an onclick event listener to the button
submitBtn.onclick = function() {
    // Get the number inputs
    var numComponentsInput = document.getElementById('Components');
    var varianceThresholdInput = document.getElementById('var');

    // Get the values of the number inputs
    var numComponents = numComponentsInput.value;
    var varianceThreshold = varianceThresholdInput.value;

    // Log the values
    console.log('Number of components:', numComponents);
    console.log('Variance threshold:', varianceThreshold);
    console.log('Selected option:', selectedOption);

    // TODO: Add your code to implement Eigen faces here
    fetch('/upload', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ numComponents: numComponents, varianceThreshold:varianceThreshold,selectedOption:selectedOption })
    })
    .then(response => response.blob())
    .then(image => {
        // Create a local URL for the image
        var image_url = URL.createObjectURL(image);
    
        // Set the source of an image element to the local URL
        div2.src = image_url;
    })
    .catch(error => console.error('Error:', error));
    
};