import java.io.IOException;
import java.util.Random;

public class train {
    public static void main(String[] args) throws IOException {

        String imageDir = "images";
        int numClasses  = 12;
        int numFilters  = 8;
        int kernelSize  = 5;
        int channels    = 1;

        DataLoader loader = new DataLoader(imageDir, true);
        System.out.println("图片总数: " + loader.size());

        Random rand = new Random(42);

        convNeuron[] neurons = new convNeuron[numFilters];
        for (int f = 0; f < numFilters; f++) {
            double[][][] kernel = new double[channels][kernelSize][kernelSize];
            double std = Math.sqrt(2.0 / (channels * kernelSize * kernelSize));
            for (int c = 0; c < channels; c++) {
                for (int i = 0; i < kernelSize; i++) {
                    for (int j = 0; j < kernelSize; j++) {
                        kernel[c][i][j] = rand.nextGaussian() * std;
                    }
                }
            }
            neurons[f] = new convNeuron(kernel, 0.0);
        }
        convLayer cL = new convLayer(neurons);

        activateLayer aL = new activateLayer(new ReLU());
        poolingLayer pL  = new poolingLayer(2, 2);
        flattenLayer fL  = new flattenLayer();

        loader.reset();
        double[][][] sample = loader.next();
        double[][][] convOut = cL.forward(sample);
        double[][][] reluOut = aL.forward(convOut);
        double[][][] poolOut = pL.forward(reluOut);
        double[]     flatOut = fL.forward(poolOut);
        int flatSize = flatOut.length;
        System.out.println("flatten 后维度: " + flatSize);

        double[][] denseW = new double[numClasses][flatSize];
        double std = Math.sqrt(2.0 / flatSize);
        for (int i = 0; i < numClasses; i++) {
            for (int j = 0; j < flatSize; j++) {
                denseW[i][j] = rand.nextGaussian() * std;
            }
        }
        double[] denseB = new double[numClasses];
        denseLayer dL = new denseLayer(denseW, denseB);

        activateVectorLayer avL = new activateVectorLayer(new Sigmoid());

        CNN cnn = new CNN(cL, aL, pL, fL, dL, avL);

        loader.reset();
        double totalLoss = 0;
        int idx = 0;

        while (loader.hasNext()) {
            String name = loader.currentFileName();
            double[][][] img = loader.next();
            double[] out = cnn.forward(img);

            double[] target = new double[numClasses];
            target[idx] = 1.0;

            double loss = 0;
            for (int i = 0; i < numClasses; i++) {
                double diff = out[i] - target[i];
                loss += diff * diff;
            }
            loss /= numClasses;
            totalLoss += loss;

            int pred = 0;
            for (int i = 1; i < numClasses; i++) {
                if (out[i] > out[pred]) pred = i;
            }

            System.out.printf("[%d] %s | 真实=%d 预测=%d loss=%.4f%n",
                    idx, name, idx, pred, loss);

            idx++;
        }

        System.out.printf("%n平均 loss: %.4f%n", totalLoss / idx);
    }
}
