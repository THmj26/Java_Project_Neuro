// 单元测试：用硬编码的小输入验证完整的前向+反向传播流程
// 目的：在真实图片训练前确认网络结构正确，loss 能持续下降
// 输入：4×4 矩阵，2个卷积核（单位对角核），2分类问题
public class test {
    public static void main(String[] args) {

        // 4×4 灰度图（单通道），值归一化到 [0,1]，与 train.java 中 /255 保持一致
        double[][][] input = {
                {
                        {0.0625, 0.1250, 0.1875, 0.2500},
                        {0.3125, 0.3750, 0.4375, 0.5000},
                        {0.5625, 0.6250, 0.6875, 0.7500},
                        {0.8125, 0.8750, 0.9375, 1.0000}
                }
        };

        // 2×2 对角卷积核：提取主对角方向特征
        double[][][] kernel1 = {{{1, 0}, {0, 1}}};
        // 2×2 副对角卷积核：提取副对角方向特征
        double[][][] kernel2 = {{{0, 1}, {1, 0}}};

        convNeuron cn1 = new convNeuron(kernel1, 0);
        convNeuron cn2 = new convNeuron(kernel2, 0);
        convLayer  cL  = new convLayer(new convNeuron[]{cn1, cn2});
        activateLayer aL = new activateLayer(new ReLU());
        poolingLayer  pL = new poolingLayer(2, 2);
        flattenLayer  fL = new flattenLayer();

        // 手动指定全连接层权重（flatten 后维度为 2）
        double[][] denseWeights = {{0.1, 0.2}, {0.8, 0.7}};
        double[]   denseBias    = {0.0, 0.0};
        denseLayer          dL  = new denseLayer(denseWeights, denseBias);
        activateVectorLayer avL = new activateVectorLayer(new Sigmoid());

        CNN cnn = new CNN(cL, aL, pL, fL, dL, avL);

        // 目标：第0类概率=1，第1类概率=0
        double[] target       = {1.0, 0.0};
        double   learningRate = 0.1;
        int      epochs       = 10000;

        // 训练循环：每100轮打印一次 loss 和输出，观察 loss 是否持续下降
        for (int epoch = 0; epoch < epochs; epoch++) {
            double[] output   = cnn.forward(input);
            double   loss     = mseLoss(output, target);
            double[] gradient = mseGradient(output, target);
            cnn.backward(gradient, learningRate);

            if (epoch % 100 == 0) {
                System.out.printf("epoch=%d  loss=%.6f  out=[%.4f, %.4f]%n",
                        epoch, loss, output[0], output[1]);
            }
        }

        System.out.println("\nfinal output:");
        for (double v : cnn.forward(input)) System.out.printf("%.4f%n", v);
    }

    // MSE = (1/2) * Σ(output[i] - target[i])²，系数1/2使导数形式简洁
    private static double mseLoss(double[] output, double[] target) {
        if (output.length != target.length) throw new IllegalArgumentException("length mismatch");
        double sum = 0;
        for (int i = 0; i < output.length; i++) {
            double d = output[i] - target[i];
            sum += d * d;
        }
        return sum / 2;
    }

    // ∂MSE/∂output[i] = output[i] - target[i]，1/2与平方的2恰好约掉
    private static double[] mseGradient(double[] output, double[] target) {
        if (output.length != target.length) throw new IllegalArgumentException("length mismatch");
        double[] g = new double[output.length];
        for (int i = 0; i < output.length; i++) {
            g[i] = output[i] - target[i];
        }
        return g;
    }
}
