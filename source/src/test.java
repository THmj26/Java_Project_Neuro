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
                {0.1, 0.2},
                {0.8, 0.7}
        };

        double[] denseBias = {0.0, 0.0};

        denseLayer dL = new denseLayer(denseWeights, denseBias);

        activateVectorLayer avL = new activateVectorLayer(new Sigmoid());

        CNN cnn = new CNN(cL, aL, pL, fL, dL, avL);

        double[] target = {1.0, 0.0};
        double learningRate = 0.01;
        int epochs = 1000;

        for (int epoch = 0; epoch < epochs; epoch++) {
            double[] output = cnn.forward(input);
            double loss = mseLoss(output, target);
            double[] gradient = mseGradient(output, target);

            cnn.backward(gradient, learningRate);

            if (epoch % 100 == 0) {
                System.out.println("epoch = " + epoch + ", loss = " + loss);
                System.out.println("output[0] = " + output[0] + ", output[1] = " + output[1]);
            }
        }

        double[] finalOutput = cnn.forward(input);
        System.out.println("final output:");
        for (int i = 0; i < finalOutput.length; i++) {
            System.out.println(finalOutput[i]);
        }
    }
    private static double mseLoss(double[] output, double[] target) {
        if (output.length != target.length) {
            throw new IllegalArgumentException("output length must match target length.");
        }

        double sum = 0.0;
        for (int i = 0; i < output.length; i++) {
            double diff = output[i] - target[i];
            sum += diff * diff;
        }

        return sum / output.length;
    }

    private static double[] mseGradient(double[] output, double[] target) {
        if (output.length != target.length) {
            throw new IllegalArgumentException("output length must match target length.");
        }

        double[] gradient = new double[output.length];
        for (int i = 0; i < output.length; i++) {
            gradient[i] = 2.0 * (output[i] - target[i]) / output.length;
        }

        return gradient;
    }
}