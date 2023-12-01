function convertImageToTensor(image) {
    return new Promise((resolve, reject) => {
        if (!(image instanceof Blob)) {
            reject(new Error('Invalid image parameter'));
            return;
        }

        const reader = new FileReader();

        reader.onload = function (e) {
            const imgElement = new Image();
            imgElement.src = e.target.result;

            imgElement.onload = () => {
                // Create a canvas with a fixed size (e.g., 48x48)
                const canvas = document.createElement('canvas');
                const targetSize = 48;

                // Resize the image while maintaining aspect ratio
                const scaleFactor = Math.min(targetSize / imgElement.width, targetSize / imgElement.height);
                const newWidth = imgElement.width * scaleFactor;
                const newHeight = imgElement.height * scaleFactor;
                canvas.width = targetSize;
                canvas.height = targetSize;

                // Center the image on the canvas
                const xOffset = (targetSize - newWidth) / 2;
                const yOffset = (targetSize - newHeight) / 2;

                // Draw the image on the canvas
                const ctx = canvas.getContext('2d');
                ctx.fillStyle = 'white';
                ctx.fillRect(0, 0, canvas.width, canvas.height);
                ctx.drawImage(imgElement, xOffset, yOffset, newWidth, newHeight);

                const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
                // Ensure the image data has a length that is a multiple of 4 (RGBA)
                if (imageData.length % 4 !== 0) {
                    reject(new Error('Invalid image data format.'));
                    return;
                }
                const normalizedData = [];
                // Copy only RGB values, skipping the alpha channel
                for (let i = 0; i < imageData.length; i += 4) {
                    normalizedData.push(imageData[i] / 255.0);
                    normalizedData.push(imageData[i + 1] / 255.0);
                    normalizedData.push(imageData[i + 2] / 255.0);
                }
                
                // Assuming your model expects the input size [1, 3, 48, 48]
                // changed 1 * 1 * 48 * 48 to
                // 1 * 3 * 48 * 48
                const expectedDims = [1, 3, 48, 48];
                

                // Debugging output
                console.log('Normalized:', normalizedData.length);
                console.log('Expected:', expectedDims.reduce((a, b) => a * b, 1));

                // Ensure that the normalized data length matches the expected size
                if (normalizedData.length !== expectedDims.reduce((a, b) => a * b, 1)) {
                    reject(new Error('Input dims do not match data length.'));
                    return;
                }

                // Create a Float32Array with size 1 * 3 * 48 * 48
                const tensorData = new Float32Array(expectedDims.reduce((a, b) => a * b, 1));

                // Copy the normalized data to the tensor data, skipping the alpha channel
                for (let i = 0; i < normalizedData.length / 3; i++) {
                    for (let j = 0; j < 3; j++) {
                        tensorData[i * 3 + j] = normalizedData[i * 3 + j];
                    }
                }

                console.log('Tensor Data:', tensorData);
                const tensor = new onnx.Tensor(tensorData, 'float32', expectedDims);

                resolve(tensor);
            };
        };

        reader.onerror = function (error) {
            reject(error);
        };

        reader.readAsDataURL(image);
    });
}

// Create an ONNX Runtime session
const session = new onnx.InferenceSession();

// Load the ONNX model
session.loadModel("emotion_recognition_model.onnx")
    .then(() => {
        console.log("Model loaded successfully.");
    })
    .catch((error) => {
        console.error("Error loading the model:", error);
    });

// Function to run inference on the uploaded image
async function runInference() {
    // Get the input image from the file input
    const fileInput = document.getElementById("imageInput");
    const image = fileInput.files[0];

    try {
        // Preprocess the image (convert to tensor, resize, normalize, etc.)
        const tensor = await convertImageToTensor(image);

        // Run inference
        const outputTensor = await session.run([tensor]);
        console.log('Output Tensor:', outputTensor); // Log the outputTensor to inspect its structure

        // Process the output (interpret the result)
        const predictedEmotion = processOutput(outputTensor);

        // Display the result on the webpage
        const outputDiv = document.getElementById("output");
        outputDiv.innerHTML = `Predicted Emotion: ${predictedEmotion}`;
    } catch (error) {
        console.error("Error during inference:", error);
    }
}

// Function to preprocess the image
async function preprocessImage(image) {
    // Convert the image to a tensor (implement your logic here)
    // Example: Assuming you have a function to convert image to tensor
    const tensor = await convertImageToTensor(image);

    // Implement any required preprocessing steps

    return tensor;
}

function processOutput(outputTensor) {
    const outputMap = outputTensor.values().next().value;
    const outputArray = outputMap.data
    
    // Log the confidence scores
    console.log('Confidence Scores:', outputArray);

    const dimensions = [1, 2];
    const maxIndex = argMax(outputArray, dimensions);

    // Class names happy and sad
    const emotionClasses = ["happy", "sad"];

    // Get the predicted emotion based on the index
    const predictedEmotion = emotionClasses[maxIndex];

    return predictedEmotion;
}

// Function to find the index of the maximum value in a multidimensional array
function argMax(array, dimensions) {
    let maxIndex = 0;
    let max = Number.NEGATIVE_INFINITY;

    for (let i = 0; i < array.length; i++) {
        if (array[i] > max) {
            max = array[i];
            maxIndex = i;
        }
    }

    const multiIndex = [];
    for (let i = 0; i < dimensions.length; i++) {
        multiIndex.push(maxIndex % dimensions[i]);
        maxIndex = Math.floor(maxIndex / dimensions[i]);
    }

    return multiIndex[0];
}
