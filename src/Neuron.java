public class Neuron {
    //Parameters
    static float minWeightValue;
    static float maxWeightValue;
    float[] weights;
    float[] cache_weights;
    float gradient;
    float bias;
    float value = 0;

    //Constructors
    public Neuron(float[] weights, float bias) {
        this.weights = weights;
        this.bias = bias;
        this.cache_weights = this.weights;
        this.gradient = 0;
    }

    public Neuron(float value) {
        this.weights = null;
        this.bias = -1;
        this.cache_weights = this.weights;
        this.gradient = -1;
        this.value = value;
    }

    public static void setRangeWeight(float min, float max) {
        minWeightValue = min;
        maxWeightValue = max;
    }


    public void update_weight() {
        this.weights = this.cache_weights;
    }


}