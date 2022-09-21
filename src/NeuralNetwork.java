public class NeuralNetwork {

    // Initiate Layers
    static Layer[] layers;

    // Initiate the data needed for training
    static Training[] tDataSet;

    public static void main(String[] args) {
        // Set the Min and Max weight value for all Neurons
        //Based on sigmoid
        //Very negative inputs end up close to 0
        //Very positive inputs end up close to 1
        Neuron.setRangeWeight(-1, 1);

        // Three layers
        layers = new Layer[3];
        layers[0] = null; // Input Layer
        layers[1] = new Layer(2, 6);
        layers[2] = new Layer(6, 1);

        // Create the training data
        CreateTraining();

        System.out.println("Before training the agent");
        for (int i = 0; i < tDataSet.length; i++) {
            forward(tDataSet[i].data);
            System.out.println(layers[2].neurons[0].value);
        }

        train(2000000, 0.05f);
        System.out.println("---------------------------------");

        System.out.println("After training the agent");
        for (int i = 0; i < tDataSet.length; i++) {
            forward(tDataSet[i].data);
            System.out.println(layers[2].neurons[0].value);
        }
    }

    // Based on XOR
    public static void CreateTraining() {
        float[] input1 = new float[]{0, 0}; //0 ^ 0 = 0
        float[] input2 = new float[]{0, 1}; //0 ^ 1 = 1
        float[] input3 = new float[]{1, 0}; //1 ^ 0 = 1
        float[] input4 = new float[]{1, 1}; //1 ^ 1 = 0

        float[] expectedOutput1 = new float[]{0};
        float[] expectedOutput2 = new float[]{1};
        float[] expectedOutput3 = new float[]{1};
        float[] expectedOutput4 = new float[]{0};


        tDataSet = new Training[4];
        tDataSet[0] = new Training(input1, expectedOutput1);
        tDataSet[1] = new Training(input2, expectedOutput2);
        tDataSet[2] = new Training(input3, expectedOutput3);
        tDataSet[3] = new Training(input4, expectedOutput4);
    }

    public static void forward(float[] inputs) {
        layers[0] = new Layer(inputs);

        for (int i = 1; i < layers.length; i++) {
            for (int j = 0; j < layers[i].neurons.length; j++) {
                float sum = 0;
                for (int k = 0; k < layers[i - 1].neurons.length; k++) {
                    sum += layers[i - 1].neurons[k].value * layers[i].neurons[j].weights[k];
                }
                layers[i].neurons[j].value = MathematicalExpressions.Sigmoid(sum);
            }
        }
    }

    public static void backward(float learning_rate, Training tData) {

        int numberOfLayers = layers.length;
        int out = numberOfLayers - 1;

        // Update the output layers
        // For each output
        for (int i = 0; i < layers[out].neurons.length; i++) {
            // and for each of their weights
            float output = layers[out].neurons[i].value;
            float target = tData.expectedOutput[i];
            float derivative = output - target;
            float delta = derivative * (output * (1 - output));
            layers[out].neurons[i].gradient = delta;
            for (int j = 0; j < layers[out].neurons[i].weights.length; j++) {
                float previous_output = layers[out - 1].neurons[j].value;
                float error = delta * previous_output;
                layers[out].neurons[i].cache_weights[j] = layers[out].neurons[i].weights[j] - learning_rate * error;
            }
        }


        for (int i = out - 1; i > 0; i--) {
            for (int j = 0; j < layers[i].neurons.length; j++) {
                float output = layers[i].neurons[j].value;
                float gradient_sum = sumGradient(j, i + 1);
                float delta = (gradient_sum) * (output * (1 - output));
                layers[i].neurons[j].gradient = delta;
                for (int k = 0; k < layers[i].neurons[j].weights.length; k++) {
                    float previous_output = layers[i - 1].neurons[k].value;
                    float error = delta * previous_output;
                    layers[i].neurons[j].cache_weights[k] = layers[i].neurons[j].weights[k] - learning_rate * error;
                }
            }
        }

        for (int i = 0; i < layers.length; i++) {
            for (int j = 0; j < layers[i].neurons.length; j++) {
                layers[i].neurons[j].update_weight();
            }
        }

    }

    public static float sumGradient(int n_index, int l_index) {
        float gradient_sum = 0;
        Layer current_layer = layers[l_index];
        for (int i = 0; i < current_layer.neurons.length; i++) {
            Neuron current_neuron = current_layer.neurons[i];
            gradient_sum += current_neuron.weights[n_index] * current_neuron.gradient;
        }
        return gradient_sum;
    }

    public static void train(int training_iterations, float learning_rate) {
        for (int i = 0; i < training_iterations; i++) {
            for (int j = 0; j < tDataSet.length; j++) {
                forward(tDataSet[j].data);
                backward(learning_rate, tDataSet[j]);
            }
        }
    }
}
