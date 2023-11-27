#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <cstdlib>
#include <functional>
#include <ctime>
#include <memory>
#include <random>
#include <algorithm> 

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
    virtual NestedVector backward(const NestedVector& gradients, double learning_rate) = 0; // Added 'backward' function
    virtual ~Layer() = default;
};

// 1D & 2D Inputs

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

    NestedVector backward(const NestedVector& gradients, double learning_rate) override {
        // You can leave this function empty for InputLayer since it's not involved in backpropagation
        NestedVector grad_input;
        return grad_input;
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
            for(auto &row : weights)
                for(auto &w : row)
                    w = distribution(generator);
            for(auto &b : biases)
                b = distribution(generator);
        }
        else if (init == ZERO) {
            for(auto &row : weights)
                for(auto &w : row)
                    w = 0.0;
            for(auto &b : biases)
                b = 0.0;
        }
        else if (init == XAVIER) {
            double var = std::sqrt(2.0 / (inputSize + outputSize));
            std::normal_distribution<double> distribution(0.0, var);
            for(auto &row : weights)
                for(auto &w : row)
                    w = distribution(generator);
            for(auto &b : biases)
                b = distribution(generator);
        }
        
   
        else if (init == ONES) {
            for (auto &row : weights)
                for (auto &w : row)
                 w = 1.0;
            for (auto &b : biases)
                b = 1.0;
        }

        else if (init == HE) {
            double var = std::sqrt(2.0 / inputSize);
            std::normal_distribution<double> distribution(0.0, var);
            for(auto &row : weights)
                for(auto &w : row)
                    w = distribution(generator);
            for(auto &b : biases)
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
            for(auto &row : weights)
                for(auto &w : row)
                    w = distribution(generator);
            for(auto &b : biases)
                b = distribution(generator);
        }
        else if (init == HE_UNIFORM) {
            double limit = std::sqrt(6.0 / inputSize);
            std::uniform_real_distribution<double> distribution(-limit, limit);
            for(auto &row : weights)
                for(auto &w : row)
                    w = distribution(generator);
            for(auto &b : biases)
                b = distribution(generator);
        }
        else if (init == LECUN_NORMAL) {
            double std_dev = std::sqrt(1.0 / inputSize);
            std::normal_distribution<double> distribution(0.0, std_dev);
            for(auto &row : weights)
                for(auto &w : row)
                    w = distribution(generator);
            for(auto &b : biases)
                b = distribution(generator);
        }
        else if (init == LECUN_UNIFORM) {
            double limit = std::sqrt(3.0 / inputSize);
            std::uniform_real_distribution<double> distribution(-limit, limit);
            for(auto &row : weights)
                for(auto &w : row)
                    w = distribution(generator);
            for(auto &b : biases)
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
        case SOFTMAX:activation= softmax;break;
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

    NestedVector backward(const NestedVector& gradients, double learning_rate) override {
    NestedVector grad_input;
    grad_input.reserve(gradients.size());

    for (int i = 0; i < gradients.size(); ++i) {
        std::vector<double> grad_input_sample(inputSize, 0.0);

        for (int j = 0; j < inputSize; ++j) {
            for (int k = 0; k < outputSize; ++k) {
                grad_input_sample[j] += gradients[i][k] * weights[j][k];
                // Update weights here using the inputs (not shown in this snippet)
            }
        }

        grad_input.push_back(grad_input_sample);
    }

    // Update biases using gradients and learning_rate
    for (int j = 0; j < outputSize; ++j) {
        double bias_grad = 0.0;
        for (int k = 0; k < gradients.size(); ++k) {
            bias_grad += gradients[k][j];
        }
        biases[j] -= learning_rate * bias_grad;
    }

    return grad_input;
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

    void fit(const NestedVector& x_train, const NestedVector& y_train, int epochs, double learning_rate) {
        int num_samples = x_train.size();

        for (int epoch = 0; epoch < epochs; ++epoch) {
            double total_loss = 0.0;

            // Shuffle training data for better convergence
            std::vector<int> indices(num_samples);
            for (int i = 0; i < num_samples; ++i) {
                indices[i] = i;
            }
            std::shuffle(indices.begin(), indices.end(), std::default_random_engine(std::time(0)));

            for (int i = 0; i < num_samples; ++i) {
                int index = indices[i];
                NestedVector input_sample = {x_train[index]};
                NestedVector target_sample = {y_train[index]};

                // Forward pass
                NestedVector output = predict(input_sample);

                // Calculate loss (mean squared error for regression)
                double loss = 0.0;
                for (int j = 0; j < output[0].size(); ++j) {
                    loss += 0.5 * std::pow(target_sample[0][j] - output[0][j], 2);
                }
                total_loss += loss;

                // Backpropagation
                NestedVector grad = target_sample;
                for (int j = 0; j < output[0].size(); ++j) {
                    grad[0][j] = output[0][j] - grad[0][j];
                }

                for (int l = layers.size() - 1; l >= 0; --l) {
                    grad = layers[l]->backward(grad, learning_rate);
                }
            }

            // Print loss for this epoch
            std::cout << "Epoch " << epoch + 1 << ", Loss: " << total_loss / num_samples << std::endl;
        }
    }

    // std::vector<int> predict_classes(const NestedVector& inputs) {
    //     NestedVector probabilities = predict(inputs);

    //     // Initialize a vector to store predicted class labels
    //     std::vector<int> predicted_labels;
    //     predicted_labels.reserve(probabilities.size());

    //     for (const auto& sample : probabilities) {
    //         int predicted_class = -1;
    //         double max_prob = 0.0;

    //         // Find the class with the highest probability
    //         for (int i = 0; i < sample.size(); ++i) {
    //             if (sample[i] > max_prob) {
    //                 max_prob = sample[i];
    //                 predicted_class = i;
    //             }
    //         }

    //         // Add the predicted class label to the result
    //         predicted_labels.push_back(predicted_class);
    //     }

    //     return predicted_labels;
    // }
};

double calculate_accuracy(const std::vector<int>& predicted_labels, const std::vector<int>& true_labels) {
    if (predicted_labels.size() != true_labels.size()) {
        throw std::invalid_argument("Input vectors must have the same size");
    }

    int correct_count = 0;
    int total_count = predicted_labels.size();

    for (size_t i = 0; i < total_count; ++i) {
        if (predicted_labels[i] == true_labels[i]) {
            correct_count++;
        }
    }

    return static_cast<double>(correct_count) / total_count;
}

double calculate_rmse(const NestedVector& predicted_values, const NestedVector& true_values) {
    if (predicted_values.size() != true_values.size()) {
        throw std::invalid_argument("Input vectors must have the same size");
    }

    double sum_squared_error = 0.0;
    int total_count = predicted_values.size();

    for (size_t i = 0; i < total_count; ++i) {
        for (size_t j = 0; j < predicted_values[i].size(); ++j) {
            double error = predicted_values[i][j] - true_values[i][j];
            sum_squared_error += error * error;
        }
    }

    double mean_squared_error = sum_squared_error / total_count;
    double rmse = std::sqrt(mean_squared_error);

    return rmse;
}


int main() 
{
    NestedVector x_train_regression = {
        {1.0, 2.0, 3.0},
        {2.0, 3.0, 4.0},
        {3.0, 4.0, 5.0},
        {4.0, 5.0, 6.0},
        {5.0, 6.0, 7.0}
    };

    NestedVector y_train_regression = {
        {4.0},
        {7.0},
        {10.0},
        {13.0},
        {16.0}
    };

    Model regression_model;
    regression_model.add(new InputLayer());
    regression_model.add(new DenseLayer(3, 5, RELU, DenseLayer::XAVIER_UNIFORM));
    regression_model.add(new DenseLayer(5, 1, LINEAR, DenseLayer::XAVIER_UNIFORM));

    // Train the model on regression problem
    regression_model.fit(x_train_regression, y_train_regression, 1000, 0.01);

    // Test Case 2: Calculate RMSE for Regression
    NestedVector x_test_regression = x_train_regression;
    NestedVector predicted_values_regression = regression_model.predict(x_test_regression);

    for(const auto &sample : predicted_values_regression) {
        for(const auto &val : sample) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }

    double rmse_regression = calculate_rmse(predicted_values_regression, y_train_regression);

    std::cout << "RMSE for Regression: " << rmse_regression << std::endl;

    return 0;
}