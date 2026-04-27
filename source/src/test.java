public class test {
    public static void main(String[] args) {

        double[][][] input = {
                {
                        {1, 2, 3, 4},
                        {5, 6, 7, 8},
                        {9, 10, 11, 12},
                        {13, 14, 15, 16}
                }
        };

        double[][][] kernel1 = {
                {
                        {1, 0},
                        {0, 1}
                }
        };

        double[][][] kernel2 = {
                {
                        {0, 1},
                        {1, 0}
                }
        };

        convNeuron cn1 = new convNeuron(kernel1, 0);
        convNeuron cn2 = new convNeuron(kernel2, 0);

        convNeuron[] neurons = {cn1, cn2};

        convLayer cL = new convLayer(neurons);

        activateLayer aL = new activateLayer(new ReLU());

        poolingLayer pL = new poolingLayer(2, 2);

        flattenLayer fL = new flattenLayer();

        double[][] denseWeights = {
                {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8},
                {0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1}
        };

        double[] denseBias = {0.0, 0.0};

        denseLayer dL = new denseLayer(denseWeights, denseBias);

        activateVectorLayer avL = new activateVectorLayer(new Sigmoid());

        CNN cnn = new CNN(cL, aL, pL, fL, dL, avL);

        double[] output = cnn.forward(input);

        for (int i = 0; i < output.length; i++) {
            System.out.println(output[i]);
        }
    }
}