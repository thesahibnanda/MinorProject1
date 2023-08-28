// Headers (Headers Are Files That Can Be Imported)
#include <iostream>
#include <vector>


class InputLayer { // Class Definition

    // Private Members Of Class
    private:
        std::vector<double> inputs;  // Vector to hold the input neurons
        int numInputs;  // Number of input neurons
    // Public Members
    public:
        // Constructor
        InputLayer(int numInputs) : numInputs(numInputs) {
            inputs.resize(numInputs, 0.0);  // Initialize with zeros
        }

        // Method to set the input values
        void setInputs(const std::vector<double>& newInputs) {
            if (newInputs.size() != numInputs) {
                std::cerr << "Error: Mismatch in number of input neurons." << std::endl;
                return;
            }
            inputs = newInputs;
        }

        // Method to get the input values
        std::vector<double> getInputs() const {
            return inputs;
        }

        // Method to print the input values
        void printInputs() const {
            std::cout << "Inputs: ";
            for (const auto& input : inputs) {
                std::cout << input << " ";
            }
            std::cout << std::endl;
        }
};


// Test the InputLayer class
int main() {
    // Create an InputLayer with 5 neurons
    InputLayer inputLayer(5);

    // Set some sample inputs
    std::vector<double> sampleInputs = {1.0, 2.0, 3.0, 4.0, 5.0};
    inputLayer.setInputs(sampleInputs);

    // Get and print the inputs
    inputLayer.printInputs();

    // Return Statement
    return 0;
}