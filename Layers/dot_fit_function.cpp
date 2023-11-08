// This Is An Example Implementation For `.fit` Function (Ignore Errors) -> Read Comments To Understand Functionality
// दिवाली की छुट्टियाँ समाप्त होने से पहले हम इसे मुख्य कोड में जोड़ देंगे


#include "Layers.cpp" // Inclusion Of Previous File (`Layers.cpp`)

class Model {
private:
    // Compute Mean Squared Error (MSE) Loss
    double computeMSE(const NestedVector& predicted, const NestedVector& target) {
        // Implement The MSE Loss Computation Here
    }

    // Compute Error For Backpropagation
    NestedVector computeError(const NestedVector& output, const NestedVector& target) {
        // Implement The Error Computation Here
    }

    // Backpropagation (Delta Rule In This Example)
    NestedVector backpropagate(Layer* layer, const NestedVector& error, double learningRate) {
        // Implement The Backpropagation Here
    }

public:

    void fit(const NestedVector& inputs, const NestedVector& targets, int epochs, double learningRate) {
        for (int epoch = 0; epoch < epochs; ++epoch) {
            double totalLoss = 0.0;

            for (size_t i = 0; i < inputs.size(); ++i) {
                // Forward Pass
                NestedVector output = inputs;
                for (const auto& layer : layers) {
                    output = layer->forward(output);
                }

                // Compute Loss (MSE In This Example)
                double loss = computeMSE(output, targets[i]);
                totalLoss += loss;

                // Backpropagation (Delta Rule In This Example)
                NestedVector error = computeError(output, targets[i]);
                for (int l = layers.size() - 1; l >= 0; --l) {
                    error = backpropagate(layer, error, learningRate);
                }
            }

            // Print Average Loss For The Epoch
            double averageLoss = totalLoss / inputs.size();
            std::cout << "Epoch " << epoch + 1 << ", Loss: " << averageLoss << std::endl;
        }
    }
};

int main()
{
    std::cout<<"Sahib Nanda";
    return 0;
}