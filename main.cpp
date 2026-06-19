#include <iostream>
#include <cmath>
#include "weights.h" // Automatically imports W0, B0, W1, B1, W2, B2

//Define the Activation Function (ReLU)
float relu(const float x) {
    return x > 0 ? x : 0;
}

// Engine
float predict_sine(float input_x) {

    // --- Layer 0: Input (1) to Hidden (16) ---
    float hidden1[16];
    for (int i = 0; i < 16; i++) {
        hidden1[i] = (input_x * W0[0][i]) + B0[i];
        hidden1[i] = relu(hidden1[i]);
    }

    // --- Layer 1: Hidden (16) to Hidden (16) ---
    float hidden2[16];
    for (int i = 0; i < 16; i++) {
        hidden2[i] = 0;
        for (int j = 0; j < 16; j++) {
            hidden2[i] += hidden1[j] * W1[j][i];
        }
        hidden2[i] += B1[i];
        hidden2[i] = relu(hidden2[i]);
    }

    // --- Layer 2: Hidden (16) to Output (1) ---
    float output = 0;
    for (int j = 0; j < 16; j++) {
        output += hidden2[j] * W2[j][0];
    }
    output += B2[0]; // Linear activation

    return output;
}

// Test
int main() {
    // Array of test values (roughly 0, pi/2, pi, 3pi/2, 2pi)
    float test_vals[] = {0.0f, 1.57f, 3.14f, 4.71f, 6.28f};

    std::cout << "TinyML Pure C++ Inference Engine\n";
    std::cout << "--------------------------------\n";

    for (float x : test_vals) {
        float prediction = predict_sine(x);
        float actual = std::sin(x);

        std::cout << "Input (x): " << x
                  << " | Pred: " << prediction
                  << " | Actual: " << actual << std::endl;
    }

    return 0;
}