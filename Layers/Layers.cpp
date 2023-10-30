#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <cstdlib>
#include <functional>
#include <ctime>
#include <memory>
#include <random>
#include <cmath>

#define _USE_MATH_DEFINES

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using ActFunc = std::function<double(double)>;
using NestedVector = std::vector<std::vector<double>>;

//**ANANYA BUNDELA** Added Softmax In enum Activation
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

// **ANANYA BUNDELA** Softmax
//START
double softmax(double x) {
    return std::exp(x) / (1.0 + std::exp(x));
}
//END

class Layer {
public:
    virtual NestedVector forward(const NestedVector& inputs) = 0;
    virtual ~Layer() = default;
};

// **AKSHITI AGARWAL** 
//START
class InputLayer : public Layer {
public:
    NestedVector forward(const NestedVector& inputs) override {
        return inputs; // Just return the input as output

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
        
        
    }
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

};
//END 

class DenseLayer : public Layer {
    int inputSize, outputSize;
    NestedVector weights;
    std::vector<double> biases;
    ActFunc activation;

public:
    enum WeightInit {
    RANDOM, ZERO, XAVIER_UNIFORM, XAVIER, HE_UNIFORM, HE, LECUN_NORMAL, LECUN_UNIFORM, MANUAL
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
        case SOFTMAX:activation= softmax;break;//**ANANYA BUNDELA** Added Softmax
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
        {4, 6, 8, 8, 9}
    };
    
    NestedVector manualWeights = {
        {0.1, 0.2, 0.3},
        {0.4, 0.5, 0.6},
        {0.7, 0.8, 0.9}
    };
    std::vector<double> manualBiases = {0.01, 0.01, 0};
    
//**AKSHITI AGARWAL**
    //START
    InputLayer inputLayer1(5);

    // For 1D input
    std::vector<double> sampleInputs_1D = {1.0, 2.0, 3.0, 4.0, 5.0};
    inputLayer1.setInputs(sampleInputs_1D);
    inputLayer1.printInputs();

    // For 2D input

    InputLayer inputLayer2(5);
    
    NestedVector sampleInputs_2D = {
        {1.0, 2.0, 3.0, 4.0, 5.0},
        {5.0, 4.0, 3.0, 2.0, 1.0},
        {1.1, 2.1, 3.1, 4.1, 5.1},
        {5.1, 4.1, 3.1, 2.1, 1.1},
        {0.0, 0.0, 0.0, 0.0, 0.0}
    };
    inputLayer2.setInputs(sampleInputs_2D);
    inputLayer2.printInputs();
    //END
    
    Model model;
    model.add(new InputLayer());
    model.add(new DenseLayer(5, 3, RELU, DenseLayer::XAVIER_UNIFORM)); // Xavier initialization
    model.add(new DenseLayer(3, 3, TANH, DenseLayer::MANUAL, 0.01, 1.0, manualWeights, manualBiases)); // Manual initialization
    model.add(new DenseLayer(3, 5, TANH, DenseLayer::RANDOM)); // Random
    model.add(new DenseLayer(5, 2, SIGMOID, DenseLayer::HE));
    model.add(new DenseLayer(2, 10, SIGMOID, DenseLayer::HE_UNIFORM));
    model.add(new DenseLayer(10, 2, SQUARE, DenseLayer::LECUN_NORMAL));
    model.add(new DenseLayer(2, 11, CUBIC, DenseLayer::LECUN_UNIFORM));
    model.add(new DenseLayer(11, 12, SQUARE_ROOT, DenseLayer::XAVIER));
    model.add(new DenseLayer(12, 1, SOFTMAX, DenseLayer::RANDOM)); // Softmax
// Manual initialization

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
