    #include <stdio.h>
    #include <stdlib.h>
    #include <math.h>
    #include <time.h>

    #define INPUT_SIZE 2
    #define HIDDEN_SIZE 100
    #define OUTPUT_SIZE 1

    typedef struct {
        double weights_input_hidden[INPUT_SIZE][HIDDEN_SIZE];
        double weights_hidden_output[HIDDEN_SIZE][OUTPUT_SIZE];
        double biases_hidden[HIDDEN_SIZE];
        double biases_output[OUTPUT_SIZE];
    } NeuralNetwork;

    double tanh_activation(double x) {
        return tanh(x);
    }

    void initialize_network(NeuralNetwork* nn) {
        srand((unsigned int)time(NULL));
        for (int i = 0; i < INPUT_SIZE; ++i)
            for (int j = 0; j < HIDDEN_SIZE; ++j)
                nn->weights_input_hidden[i][j] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
        for (int i = 0; i < HIDDEN_SIZE; ++i)
            for (int j = 0; j < OUTPUT_SIZE; ++j)
                nn->weights_hidden_output[i][j] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
        for (int i = 0; i < HIDDEN_SIZE; ++i)
            nn->biases_hidden[i] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
        for (int i = 0; i < OUTPUT_SIZE; ++i)
            nn->biases_output[i] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
    }

    void feedforward(NeuralNetwork* nn, double input[INPUT_SIZE], double hidden[HIDDEN_SIZE], double output[OUTPUT_SIZE]) {
        for (int i = 0; i < HIDDEN_SIZE; ++i) {
            hidden[i] = nn->biases_hidden[i];
            for (int j = 0; j < INPUT_SIZE; ++j)
                hidden[i] += input[j] * nn->weights_input_hidden[j][i];
            hidden[i] = tanh_activation(hidden[i]);
        }
        for (int i = 0; i < OUTPUT_SIZE; ++i) {
            output[i] = nn->biases_output[i];
            for (int j = 0; j < HIDDEN_SIZE; ++j)
                output[i] += hidden[j] * nn->weights_hidden_output[j][i];
            output[i] = tanh_activation(output[i]);
        }
    }

    void train(NeuralNetwork* nn, double input[INPUT_SIZE], double target[OUTPUT_SIZE], double learning_rate) {
        double hidden[HIDDEN_SIZE], output[OUTPUT_SIZE];
        feedforward(nn, input, hidden, output);

        double output_errors[OUTPUT_SIZE];
        for (int i = 0; i < OUTPUT_SIZE; ++i)
            output_errors[i] = target[i] - output[i];

        for (int i = 0; i < OUTPUT_SIZE; ++i) {
            for (int j = 0; j < HIDDEN_SIZE; ++j)
                nn->weights_hidden_output[j][i] += learning_rate * output_errors[i] * (1 - output[i] * output[i]) * hidden[j];
            nn->biases_output[i] += learning_rate * output_errors[i] * (1 - output[i] * output[i]);
        }

        for (int i = 0; i < HIDDEN_SIZE; ++i) {
            double hidden_error = 0.0;
            for (int j = 0; j < OUTPUT_SIZE; ++j)
                hidden_error += output_errors[j] * nn->weights_hidden_output[i][j];
            for (int j = 0; j < INPUT_SIZE; ++j)
                nn->weights_input_hidden[j][i] += learning_rate * hidden_error * (1 - hidden[i] * hidden[i]) * input[j];
            nn->biases_hidden[i] += learning_rate * hidden_error * (1 - hidden[i] * hidden[i]);
        }
    }

    int main() {
        NeuralNetwork nn;
        initialize_network(&nn);

        double input[INPUT_SIZE] = {0.5, 0.3};
        double target[OUTPUT_SIZE] = {0.8};
        double learning_rate = 0.00001;
        double error_threshold = 0.01;
        double error = 1.0;

        double hidden[HIDDEN_SIZE], output[OUTPUT_SIZE];

        while (error > error_threshold) {
            train(&nn, input, target, learning_rate);
            feedforward(&nn, input, hidden, output);
            error = fabs(target[0] - output[0]);
            printf("Current output: %.6f, Error: %.6f\n", output[0], error);
        }
        printf("The training is finally complete. Here is the output: %.6f\n", output[0]);
        return 0;
    }