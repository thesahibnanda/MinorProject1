#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <cstdlib>
#include <functional>
#include <ctime>
#include <memory>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using ActFunc = std::function<double(double)>;
using NestedVector = std::vector<std::vector<double>>;

enum Activation {
    RELU, LEAKY_RELU, PARAMETRIC_RELU, SWISH, EXPONENTIAL, TANH, LINEAR, GELU, SIGMOID, NONE
};

double relu(double x) {
    return x > 0 ? x : 0;
}

double leaky_relu(double x, double alpha = 0.01) {
    return x > 0 ? x : alpha * x;
}

double parametric_relu(double x, double alpha) {
    return x > 0 ? x : alpha * x;
}

double swish(double x, double beta = 1.0) {
    return x / (1.0 + std::exp(-beta * x));
}

double exponential(double x) {
    return std::exp(x);
}

double tanh_act(double x) {
    return std::tanh(x);
}

double gelu(double x) {
    return 0.5 * x * (1 + std::tanh(std::sqrt(2 / M_PI) * (x + 0.044715 * std::pow(x, 3))));
}

double sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

class Layer {
public:
    virtual NestedVector forward(const NestedVector& inputs) = 0;
    virtual ~Layer() {}
};

class InputLayer : public Layer {
public:
    NestedVector forward(const NestedVector& inputs) override {
        return inputs;
    }
};

class DenseLayer : public Layer {
    int inputSize;
    int outputSize;
    NestedVector weights;
    std::vector<double> biases;
    ActFunc activation;

public:
    DenseLayer(int inputSize, int outputSize, Activation act = NONE, double alpha = 0.01, double beta = 1.0) : 
    inputSize(inputSize), outputSize(outputSize) {
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
        case LEAKY_RELU: activation = [alpha](double x) { return leaky_relu(x, alpha); }; break;
        case PARAMETRIC_RELU: activation = [alpha](double x) { return parametric_relu(x, alpha); }; break;
        case SWISH: activation = [beta](double x) { return swish(x, beta); }; break;
        case EXPONENTIAL: activation = exponential; break;
        case TANH: activation = tanh_act; break;
        case LINEAR: activation = [](double x) { return x; }; break;
        case GELU: activation = gelu; break;
        case SIGMOID: activation = sigmoid; break;
        case NONE: activation = [](double x) { return x; }; break;
        default: throw std::invalid_argument("Unsupported activation function");
        }
    }

    NestedVector forward(const NestedVector& inputs) override {
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
    std::vector<std::unique_ptr<Layer>> layers;

public:
    void add(Layer* layer) {
        layers.emplace_back(layer);
    }

    NestedVector predict(const NestedVector& inputs) {
        NestedVector currentOutput = inputs;
        for (const auto& layer : layers) {
            currentOutput = layer->forward(currentOutput);
        }
        return currentOutput;
    }
};

int main() {
    NestedVector sampleInputs = {
        {1, 2, 3, 4, 5},
        {5, 4, 3, 2, 1},
        {4, 6, 8, 8, 9},
        {1, 4, 4, 5, 6}
    };
    
    Model model;
    model.add(new InputLayer());
    model.add(new DenseLayer(5, 3, RELU));
    model.add(new DenseLayer(3, 3, SIGMOID));
    model.add(new DenseLayer(3, 10, GELU));
    model.add(new DenseLayer(10, 1, LEAKY_RELU));

    NestedVector x_test = sampleInputs;

    NestedVector y_pred = model.predict(x_test);
    
    std::cout << "Prediction Results:" << std::endl;
    for(const auto &sample : y_pred) {
        for(const auto &val : sample) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}