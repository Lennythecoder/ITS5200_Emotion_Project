// Function to process the output tensor
function processOutput(outputTensor) {
    // Convert the Float32Array to a regular array
    const outputArray = Array.from(outputTensor);

    const dimensions = [1, 2];
    const maxIndex = argMax(outputArray, dimensions);

    // Class names happy and sad
    const emotionClasses = ["Happy", "Sad"];

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

    return multiIndex[1];
}
