#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <cstdlib>
#include <ctime>

using NestedVector = std::vector<std::vector<double>>; // For 2D inputs

enum Activation {RELU, SIGMOID, TANH, NONE};

double relu(double x) {
    return x > 0 ? x : 0;
}

double sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

double tanh_act(double x) {
    return std::tanh(x);
}

// Function pointer type for activation functions
using ActFunc = double(*)(double);

class Layer {
public:
    virtual NestedVector forward(const NestedVector& inputs) = 0;
};

class InputLayer : public Layer {
private:
    NestedVector inputs;  // For 2D inputs
    std::vector<double> inputs_1D; // For 1D inputs
    int numInputs;
    bool is1D; // A flag to indicate if the input is 1D or not

    bool checkNestedVectorSize(const NestedVector& vec, int expectedSize) {
        if (vec.size() != expectedSize) return false;
        for (const auto& subVec : vec) {
            if (subVec.size() != expectedSize) return false;
        }
        return true;
    }

public:
    InputLayer(int numInputs) : numInputs(numInputs), is1D(true) {
        inputs_1D.resize(numInputs, 0.0);
    }

    // For 1D input
    void setInputs(const std::vector<double>& newInputs) {
        if (newInputs.size() != numInputs) {
            std::cerr << "Error: Mismatch in number of input neurons." << std::endl;
            return;
        }
        inputs_1D = newInputs;
        is1D = true; // set the flag to indicate it's 1D input
    }

    // For 2D input
    void setInputs(const NestedVector& newInputs) {
        if (!checkNestedVectorSize(newInputs, numInputs)) {
            std::cerr << "Error: Mismatch in number of input neurons." << std::endl;
            return;
        }
        inputs = newInputs;
        is1D = false; // reset the flag to indicate it's not 1D input
    }

    void printInputs() const {
        if (is1D) {
            std::cout << "1D Inputs: ";
            for (const auto& input : inputs_1D) {
                std::cout << input << " ";
            }
            std::cout << std::endl;
        } else {
            std::cout << "2D Inputs: " << std::endl;
            for (const auto& row : inputs) {
                for (const auto& input : row) {
                    std::cout << input << " ";
                }
                std::cout << std::endl;
            }
        }
    }

    // Forward method for compatibility with other layers
    virtual NestedVector forward(const NestedVector& newInputs) override {
        if (is1D) {
            return {inputs_1D};
        }
        return inputs;
    }
};


class DenseLayer : public Layer {
private:
    NestedVector weights;
    std::vector<double> biases;
    ActFunc activation;
    int inputSize;
    int outputSize;

public:
    DenseLayer(int inputSize, int outputSize, Activation act = NONE) : 
        inputSize(inputSize), outputSize(outputSize) {
        
        // Initialize weights and biases to random values for demonstration purposes
        std::srand(std::time(0));
        weights.resize(inputSize, std::vector<double>(outputSize));
        for(auto &row : weights) {
            for(auto &w : row) {
                w = ((double) rand() / (RAND_MAX)) - 0.5;
            }
        }
        biases.resize(outputSize);
        for(auto &b : biases) {
            b = ((double) rand() / (RAND_MAX)) - 0.5;
        }
        
        switch (act) {
            case RELU: activation = relu; break;
            case SIGMOID: activation = sigmoid; break;
            case TANH: activation = tanh_act; break;
            case NONE: activation = [](double x) -> double { return x; }; break;
            default: throw std::invalid_argument("Unsupported activation function");
        }
    }

    NestedVector forward(const NestedVector& inputs) override {
        // Perform the forward computation here
        // You can assume that 'inputs' is a 2D vector (each element is a 1D vector representing a sample)
        NestedVector output;

        for (const auto& sample : inputs) {
            if (sample.size() != inputSize) {
                throw std::invalid_argument("Input size mismatch");
            }

            std::vector<double> outputSample(outputSize, 0.0);
            for (int o = 0; o < outputSize; ++o) {
                for (int i = 0; i < inputSize; ++i) {
                    outputSample[o] += sample[i] * weights[i][o];
                }
                outputSample[o] += biases[o];
                outputSample[o] = activation(outputSample[o]);
            }
            output.push_back(outputSample);
        }

        return output;
    }
};

class Model {
private:
    std::vector<Layer*> layers;

public:
    void add(Layer* layer) {
        layers.push_back(layer);
    }

    NestedVector predict(const NestedVector& inputs) {
        NestedVector currentOutput = inputs;
        for (Layer* layer : layers) {
            currentOutput = layer->forward(currentOutput);
        }
        return currentOutput;
    }
};

// Usage
int main() {
    // Same usage for InputLayer as in your code
    InputLayer inputLayer(5);
    NestedVector sampleInputs = {
        {1, 2, 3, 4, 5},
        {5, 4, 3, 2, 1},
        {4, 6, 8, 8, 9}
    };
    
    Model model;
    model.add(&inputLayer);
    model.add(new DenseLayer(5, 3, RELU));
    model.add(new DenseLayer(3, 1, SIGMOID));

    // Assume x_test is a 2D array of test samples
    NestedVector x_test = sampleInputs;

    NestedVector y_pred = model.predict(x_test);
    
    // Output y_pred
    std::cout << "Prediction Results:" << std::endl;
    for(const auto &sample : y_pred) {
        for(const auto &val : sample) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
    return 0;
}
