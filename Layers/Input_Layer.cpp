#include <iostream>
#include <vector>

using NestedVector = std::vector<std::vector<double>>; // For 2D inputs

class InputLayer {
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
};

int main() {
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

    return 0;
}
