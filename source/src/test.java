public class test {
    static void main() {
        double[][][] input = {{
                {1, 2, 3, 0, 1},
                {0, 1, 2, 3, 1},
                {1, 0, 1, 2, 2},
                {2, 1, 0, 1, 3},
                {1, 2, 1, 0, 1}
        }};

        double[][][] kernel = {{
                {1, 0, -1},
                {1, 0, -1},
                {1, 0, -1}
        }};
        convNeuron conv = new convNeuron(kernel, 0.0);
        double[][] output = conv.forward(input);

        for (double[] doubles : output) {
            for (int j = 0; j < output[0].length; j++) {
                System.out.print(doubles[j] + " ");
            }
            System.out.println();
        }
    }
}
