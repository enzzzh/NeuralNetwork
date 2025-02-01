// copyright enzzz 2024
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <iomanip>
#include <chrono>
#include <thread>

class NeuralNetwork {
public:
    NeuralNetwork(int input_size, int hidden_size, int output_size)
        : input_size(input_size), hidden_size(hidden_size), output_size(output_size) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dis(-1.0, 1.0);

        // Initialize weights
        weights_input_hidden.resize(input_size, std::vector<double>(hidden_size));
        weights_hidden_output.resize(hidden_size, std::vector<double>(output_size));

        for (auto& row : weights_input_hidden)
            for (auto& w : row)
                w = dis(gen);

        for (auto& row : weights_hidden_output)
            for (auto& w : row)
                w = dis(gen);

        // Initialize biases
        biases_hidden.resize(hidden_size);
        biases_output.resize(output_size);
        for (auto& b : biases_hidden)
            b = dis(gen);
        for (auto& b : biases_output)
            b = dis(gen);
    }

    std::vector<double> feedforward(const std::vector<double>& input) {
        std::vector<double> hidden(hidden_size);
        for (int i = 0; i < hidden_size; ++i) {
            hidden[i] = biases_hidden[i];
            for (int j = 0; j < input_size; ++j) {
                hidden[i] += input[j] * weights_input_hidden[j][i];
            }
            hidden[i] = tanh(hidden[i]); 
        }

        // Hidden to output layer
        std::vector<double> output(output_size);
        for (int i = 0; i < output_size; ++i) {
            output[i] = biases_output[i];
            for (int j = 0; j < hidden_size; ++j) {
                output[i] += hidden[j] * weights_hidden_output[j][i];
            }
            output[i] = tanh(output[i]); 
        }
        return output;
    }

    void train(const std::vector<double>& input, const std::vector<double>& target, double learning_rate) {
        // Feedforward
        std::vector<double> hidden(hidden_size);
        for (int i = 0; i < hidden_size; ++i) {
            hidden[i] = biases_hidden[i];
            for (int j = 0; j < input_size; ++j) {
                hidden[i] += input[j] * weights_input_hidden[j][i];
            }
            hidden[i] = tanh(hidden[i]);
        }

        std::vector<double> output(output_size);
        for (int i = 0; i < output_size; ++i) {
            output[i] = biases_output[i];
            for (int j = 0; j < hidden_size; ++j) {
                output[i] += hidden[j] * weights_hidden_output[j][i];
            }
            output[i] = tanh(output[i]);
        }

        // Calculate output error
        std::vector<double> output_errors(output_size);
        for (int i = 0; i < output_size; ++i) {
            output_errors[i] = target[i] - output[i];
        }

        // Update weights and biases 
        for (int i = 0; i < output_size; ++i) {
            for ( int j = 0; j < hidden_size; ++j) {
                weights_hidden_output[j][i] += learning_rate * output_errors[i] * (1 - output[i] * output[i]) * hidden[j];
            }
            biases_output[i] += learning_rate * output_errors[i] * (1 - output[i] * output[i]);
        }

        for (int i = 0; i < hidden_size; ++i) {
            double hidden_error = 0.0;
            for (int j = 0; j < output_size; ++j) {
                hidden_error += output_errors[j] * weights_hidden_output[i][j];
            }
            for (int j = 0; j < input_size; ++j) {
                weights_input_hidden[j][i] += learning_rate * hidden_error * (1 - hidden[i] * hidden[i]) * input[j];
            }
            biases_hidden[i] += learning_rate * hidden_error * (1 - hidden[i] * hidden[i]);
        }
    }

private:
    int input_size;
    int hidden_size;
    int output_size;

    std::vector<std::vector<double>> weights_input_hidden;
    std::vector<std::vector<double>> weights_hidden_output;
    std::vector<double> biases_hidden;
    std::vector<double> biases_output;

    double sigmoid(double x) {
        return 1.0 / (1.0 + exp(-x));
    }
};

int main() {
    NeuralNetwork nn(2, 3, 1); // 2 input neurons, 3 hidden neurons, 1 output neuron

    std::vector<double> input = {0.5, 0.3};
    std::vector<double> target = {0.8}; // Desired Output 
    double learning_rate = 0.1;
    double error_threshold = 0.01;

    double error = 1.0; // Initial error for loop 

    while (error > error_threshold) {
        nn.train(input, target, learning_rate);


        std::vector<double> output = nn.feedforward(input);
        error = std::abs(target[0] - output[0]); // Calculate error

        std::cout << std::fixed << std::setprecision(6) << "Current output: " << output[0] << ", Error: " << error << std::endl;

    }
    std::cout << "The training is finally complete. Here is the output: " << nn.feedforward(input)[0] << std::endl;
    return 0;
}