import tempfile
import os
import numpy as np
import json
import tensorflow as tf
from tensorflow import keras
import tensorflow_model_optimization as tfmot

# Load MNIST dataset
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the input image so that each pixel value is between 0 to 1.
train_images = train_images / 255.0
test_images = test_images / 255.0

# Define the model architecture.
model = keras.Sequential([
  keras.layers.InputLayer(input_shape=(28, 28)),
  keras.layers.Reshape(target_shape=(28, 28, 1)),
  keras.layers.Conv2D(filters=12, kernel_size=(3, 3), activation='relu'),
  keras.layers.MaxPooling2D(pool_size=(2, 2)),
  keras.layers.Flatten(),
  keras.layers.Dense(10)
])

# Train the digit classification model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(
  train_images,
  train_labels,
  epochs=1,
  validation_split=0.1,
)

quantize_model = tfmot.quantization.keras.quantize_model

# Quantization-aware training
q_aware_model = quantize_model(model)
q_aware_model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

q_aware_model.summary()

train_images_subset = train_images[0:1000]
train_labels_subset = train_labels[0:1000]

q_aware_model.fit(train_images_subset, train_labels_subset,
                  batch_size=500, epochs=1, validation_split=0.1)

_, baseline_model_accuracy = model.evaluate(
    test_images, test_labels, verbose=0)

_, q_aware_model_accuracy = q_aware_model.evaluate(
   test_images, test_labels, verbose=0)

print('Baseline test accuracy:', baseline_model_accuracy)
print('Quant test accuracy:', q_aware_model_accuracy)

results = {
    "baseline_model_accuracy_before_injection": baseline_model_accuracy,
    "q_aware_model_accuracy_before_injection": q_aware_model_accuracy,
    "baseline_model_accuracy_after_injection": [],
    "q_aware_model_accuracy_after_injection": [],
    "ber_values": []  # New key to store BER values
}

def inject_bit_errors(weights, bit_error_rate):
    modified_weights = []
    for weight_array in weights:
        flat_weights = weight_array.flatten()
        num_bits_to_flip = int(bit_error_rate * len(flat_weights))
        bit_indices = np.random.choice(len(flat_weights), size=num_bits_to_flip, replace=False)
        for idx in bit_indices:
            flat_weights[idx] = 1 - flat_weights[idx]
        modified_weights.append(flat_weights.reshape(weight_array.shape))
    return modified_weights

# BER values
bit_error_rates = np.logspace(-10, -2, num=1000)

for ber in bit_error_rates:
    results["ber_values"].append(ber)  # Append BER value to the results dictionary

    # Inject bit errors into the weights of the original model
    modified_weights = inject_bit_errors(model.get_weights(), bit_error_rate=ber)
    model.set_weights(modified_weights)

    # Inject bit errors into the weights of the quantization-aware model
    modified_q_aware_weights = inject_bit_errors(q_aware_model.get_weights(), bit_error_rate=ber)
    q_aware_model.set_weights(modified_q_aware_weights)

    # Evaluate the modified models with injected bit errors
    _, modified_baseline_accuracy = model.evaluate(test_images, test_labels, verbose=0)
    _, modified_q_aware_accuracy = q_aware_model.evaluate(test_images, test_labels, verbose=0)

    results["baseline_model_accuracy_after_injection"].append(modified_baseline_accuracy)
    results["q_aware_model_accuracy_after_injection"].append(modified_q_aware_accuracy)

# Save results to JSON file
with open('accuracy_results.json', 'w') as fp:
    json.dump(results, fp)