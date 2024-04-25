document.addEventListener("DOMContentLoaded", function() {
    const canvas = document.getElementById('draw-canvas');
    const context = canvas.getContext('2d');
    let isDrawing = false;

    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', stopDrawing);
    document.getElementById('clear-btn').addEventListener('click', clearCanvas);
    document.getElementById('classify-btn').addEventListener('click', classifyDrawing);

    // Set the initial canvas background and drawing styles
    context.fillStyle = 'white';                            // Background color
    context.fillRect(0, 0, canvas.width, canvas.height);    // Apply background color
    context.strokeStyle = 'black';                          // Drawing color
    context.lineWidth = 10;                                 // Line width for drawing
    context.lineCap = 'round';                              // Line cap for smoother lines

    function startDrawing(e) {
        isDrawing = true;
        draw(e);
    }

    function draw(e) {
        if (!isDrawing) return;
        context.lineWidth = 10;  // Match the scaling of the canvas to the model input
        context.lineCap = 'round';
        context.lineTo(e.clientX - canvas.offsetLeft, e.clientY - canvas.offsetTop);
        context.stroke();
        context.beginPath();
        context.moveTo(e.clientX - canvas.offsetLeft, e.clientY - canvas.offsetTop);
    }

    function stopDrawing() {
        isDrawing = false;
        context.beginPath();
    }

    function clearCanvas() {    
        context.fillStyle = 'white'; // Ensure the background is white
        context.fillRect(0, 0, canvas.width, canvas.height);
        context.lineWidth = 10; // Ensure line width is sufficient
        context.strokeStyle = 'black'; // Ensure drawing color is black
        context.beginPath();
        }

    function classifyDrawing() {
        const imageData = canvas.toDataURL('image/png');
        console.log(imageData);  // Check what this outputs in the browser's console

        fetch('/classify', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ image_data: imageData })
        })
        .then(response => response.json())
        .then(data => {
            alert('Predicted class: ' + data.class);
        })
        .catch(error => {
            console.error('Error:', error);
        });
    }
});
