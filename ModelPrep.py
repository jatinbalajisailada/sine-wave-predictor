import numpy as np
import math
import tensorflow as tf

# Generate Dummy Data (Sine Wave)
# We generate 1000 random data points between 0 and 2*pi
x_val = np.random.uniform(0, 2 * math.pi, 1000).astype(np.float32)
np.random.shuffle(x_val)
y_val = np.sin(x_val).astype(np.float32)

# Build a Tiny Model
# 1 input -> 16 hidden (ReLU) -> 16 hidden (ReLU) -> 1 output (Linear)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(1,)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Model Training
model.compile(optimizer='adam', loss='mse')
print("Training model...")
model.fit(x_val, y_val, epochs=150, batch_size=16)
print("Training complete!")

# Extract Weights and Write to C Header File
def ExportWeightsToC(model, filename="weights.h"):
    with open(filename, "w") as f:
        f.write("#ifndef WEIGHTS_H\n#define WEIGHTS_H\n\n")
        f.write("// Auto-generated weights for TinyML Sine Wave Predictor\n\n")

        for i, layer in enumerate(model.layers):
            weights = layer.get_weights()
            if not weights:
                continue

            w, b = weights

            # Write weights array
            f.write(f"const float W{i}[{w.shape[0]}][{w.shape[1]}] = {{\n")
            for row in w:
                f.write("  {" + ", ".join(map(str, row)) + "},\n")
            f.write("};\n\n")

            # Write biases array
            f.write(f"const float B{i}[{b.shape[0]}] = {{\n")
            f.write("  " + ", ".join(map(str, b)) + "\n")
            f.write("};\n\n")
        f.write("#endif // WEIGHTS_H\n")

ExportWeightsToC(model)
print("Extracted raw weights and saved to 'weights.h'")