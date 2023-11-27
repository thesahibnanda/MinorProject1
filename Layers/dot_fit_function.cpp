#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <cstdlib>
#include <functional>
#include <ctime>
#include <memory>
#include <algorithm>
#include <random>

#define _USE_MATH_DEFINES
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using ActFunc = std::function<double(double)>;
using NestedVector = std::vector<std::vector<double>>;

enum Activation {
    RELU, LEAKY_RELU, PARAMETRIC_RELU, SWISH, EXPONENTIAL, TANH, LINEAR, GELU, SIGMOID, NONE, SQUARE, SQUARE_ROOT, CUBIC, SOFTMAX
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

double square(double x) {
    return x * x;
}

double square_root(double x) {
    if (x < 0) {
        return 0; // Square root of negative numbers is undefined in real numbers
    }
    return std::sqrt(x);
}

double cubic(double x) {
    return x * x * x;
}

double softmax(double x) {
    return std::exp(x) / (1.0 + std::exp(x));
}

class Layer {
public:
    virtual NestedVector forward(const NestedVector& inputs) = 0;
    virtual ~Layer() = default;
};

class InputLayer : public Layer {
public:
    NestedVector forward(const NestedVector& inputs) override {
        NestedVector output;
        for (const auto& sample : inputs) {
            std::vector<double> outputSample;
            for (double value : sample) {
                outputSample.push_back(value);
            }
            output.push_back(outputSample);
        }
        return output;
    }
};

class DenseLayer : public Layer {
    int inputSize, outputSize;
    NestedVector weights;
    std::vector<double> biases;
    ActFunc activation;

public:
    enum WeightInit {
        RANDOM, ZERO, XAVIER_UNIFORM, XAVIER, HE_UNIFORM, HE, LECUN_NORMAL, LECUN_UNIFORM, MANUAL, ONES
    };

    DenseLayer(int inputSize, int outputSize, Activation act = NONE, WeightInit init = RANDOM,
               double alpha = 0.01, double beta = 1.0, const NestedVector& weightData = {}, const std::vector<double>& biasData = {}) :
        inputSize(inputSize), outputSize(outputSize), weights(inputSize, std::vector<double>(outputSize)), biases(outputSize) {

        std::default_random_engine generator(std::time(0));

        if (init == RANDOM) {
            std::uniform_real_distribution<double> distribution(-0.5, 0.5);
            for (auto& row : weights)
                for (auto& w : row)
                    w = distribution(generator);
            for (auto& b : biases)
                b = distribution(generator);
        }
        else if (init == ZERO) {
            for (auto& row : weights)
                for (auto& w : row)
                    w = 0.0;
            for (auto& b : biases)
                b = 0.0;
        }
        else if (init == XAVIER) {
            double var = std::sqrt(2.0 / (inputSize + outputSize));
            std::normal_distribution<double> distribution(0.0, var);
            for (auto& row : weights)
                for (auto& w : row)
                    w = distribution(generator);
            for (auto& b : biases)
                b = distribution(generator);
        }

        else if (init == ONES) {
            for (auto& row : weights)
                for (auto& w : row)
                    w = 1.0;
            for (auto& b : biases)
                b = 1.0;
        }

        else if (init == HE) {
            double var = std::sqrt(2.0 / inputSize);
            std::normal_distribution<double> distribution(0.0, var);
            for (auto& row : weights)
                for (auto& w : row)
                    w = distribution(generator);
            for (auto& b : biases)
                b = distribution(generator);
        }
        else if (init == MANUAL) {
            if (weightData.size() != inputSize || weightData[0].size() != outputSize || biasData.size() != outputSize) {
                throw std::invalid_argument("Weight and bias data dimensions do not match layer dimensions");
            }
            weights = weightData;
            biases = biasData;
        }

        else if (init == XAVIER_UNIFORM) {
            double limit = std::sqrt(6.0 / (inputSize + outputSize));
            std::uniform_real_distribution<double> distribution(-limit, limit);
            for (auto& row : weights)
                for (auto& w : row)
                    w = distribution(generator);
            for (auto& b : biases)
                b = distribution(generator);
        }
        else if (init == HE_UNIFORM) {
            double limit = std::sqrt(6.0 / inputSize);
            std::uniform_real_distribution<double> distribution(-limit, limit);
            for (auto& row : weights)
                for (auto& w : row)
                    w = distribution(generator);
            for (auto& b : biases)
                b = distribution(generator);
        }
        else if (init == LECUN_NORMAL) {
            double std_dev = std::sqrt(1.0 / inputSize);
            std::normal_distribution<double> distribution(0.0, std_dev);
            for (auto& row : weights)
                for (auto& w : row)
                    w = distribution(generator);
            for (auto& b : biases)
                b = distribution(generator);
        }
        else if (init == LECUN_UNIFORM) {
            double limit = std::sqrt(3.0 / inputSize);
            std::uniform_real_distribution<double> distribution(-limit, limit);
            for (auto& row : weights)
                for (auto& w : row)
                    w = distribution(generator);
            for (auto& b : biases)
                b = distribution(generator);
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
        case SQUARE: activation = square; break; // Add SQUARE activation
        case SQUARE_ROOT: activation = square_root; break; // Add SQUARE_ROOT activation
        case CUBIC: activation = cubic; break; // Add CUBIC activation
        case SOFTMAX: activation = softmax; break;
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
    double learningRate; // Learning rate for gradient descent

public:
    Model(double lr = 0.01) : learningRate(lr) {}

    void add(Layer* layer) {
        layers.emplace_back(layer);
    }

    void fit(const NestedVector& X_train, const NestedVector& y_train, int epochs, int batchSize = 32) {
        if (X_train.size() != y_train.size()) {
            throw std::invalid_argument("Input and output data sizes do not match");
        }

        int numSamples = X_train.size();

        for (int epoch = 0; epoch < epochs; ++epoch) {
            // Shuffle the dataset at the beginning of each epoch (optional)
            std::vector<int> indices(numSamples);
            std::iota(indices.begin(), indices.end(), 0);
            std::shuffle(indices.begin(), indices.end(), std::default_random_engine(std::time(0)));

            for (int batchStart = 0; batchStart < numSamples; batchStart += batchSize) {
                int batchEnd = std::min(batchStart + batchSize, numSamples);

                // Prepare a batch
                NestedVector batchX;
                NestedVector batchY;
                for (int i = batchStart; i < batchEnd; ++i) {
                    batchX.push_back(X_train[indices[i]]);
                    batchY.push_back(y_train[indices[i]]);
                }

                // Forward pass
                NestedVector currentOutput = batchX;
                for (const auto& layer : layers) {
                    currentOutput = layer->forward(currentOutput);
                }

                // Calculate loss and gradients
                // For regression, you can use Mean Squared Error (MSE) as the loss
                // For classification, you can use Cross-Entropy Loss
                // Calculate gradients using backpropagation

                // Update weights and biases using gradient descent
                // w_new = w_old - learningRate * gradient
                // b_new = b_old - learningRate * gradient
            }
        }
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
    // ... (Your existing code here)

    // Regression Example
    NestedVector X_train_reg = {{1.0, 2.0}, {2.0, 3.0}, {3.0, 4.0}};
    NestedVector y_train_reg = {{3.0}, {5.0}, {7.0}};

    Model regressionModel; // Use the default learning rate (0.01)
    Model classificationModel(0.01); // Provide a custom learning rate

    regressionModel.add(new InputLayer());
    regressionModel.add(new DenseLayer(2, 1, LINEAR, DenseLayer::RANDOM));

    regressionModel.fit(X_train_reg, y_train_reg, 1000);

    NestedVector x_test_reg = {{4.0, 5.0}, {5.0, 6.0}};
    NestedVector y_pred_reg = regressionModel.predict(x_test_reg);

    std::cout << "Regression Results:" << std::endl;
    for (const auto& sample : y_pred_reg) {
        for (const auto& val : sample) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }

    // Classification Example
    NestedVector X_train_cls = {{1.0, 2.0}, {2.0, 3.0}, {3.0, 4.0}, {1.0, 1.0}, {2.0, 2.0}, {3.0, 3.0}};
    NestedVector y_train_cls = {{0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}};

    classificationModel.add(new InputLayer());
    classificationModel.add(new DenseLayer(2, 2, SIGMOID, DenseLayer::RANDOM));

    classificationModel.fit(X_train_cls, y_train_cls, 1000);

    NestedVector x_test_cls = {{4.0, 5.0}, {1.0, 2.0}};
    NestedVector y_pred_cls = classificationModel.predict(x_test_cls);

    std::cout << "Classification Results:" << std::endl;
    for (const auto& sample : y_pred_cls) {
        for (const auto& val : sample) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}