import java.io.File;
import java.io.IOException;
import java.util.Random;

// 主训练脚本：加载十二生肖图片，初始化 CNN，执行多轮前向+反向传播训练
// 流程：初始化权重 → 动态推导 flatten 维度 → 训练循环（前向/MSE/反向/更新）→ 测试报告
public class train {
    public static void main(String[] args) throws IOException {

        // ── 超参数 ──────────────────────────────────────────────────
        String imageDir     = "images"; // 图片目录（64×64 灰度图）
        int numClasses      = 12;       // 输出类别数：十二生肖
        int numFilters      = 8;        // 卷积层 filter 数量
        int kernelSize      = 5;        // 卷积核大小：5×5
        int channels        = 1;        // 输入通道数：灰度图为 1
        int epochs          = 200;      // 训练轮数
        double learningRate = 0.001;    // SGD 学习率

        // ── 1. 加载图片 ──────────────────────────────────────────────
        DataLoader loader = new DataLoader(imageDir, true);
        System.out.println("图片总数: " + loader.size());

        // ── 2. 初始化卷积层 ──────────────────────────────────────────
        // He 初始化：权重 ~ N(0, sqrt(2 / fan_in))，适合 ReLU 激活，防止初始梯度消失/爆炸
        Random rand = new Random(42);
        convNeuron[] neurons = new convNeuron[numFilters];
        for (int f = 0; f < numFilters; f++) {
            double[][][] kernel = new double[channels][kernelSize][kernelSize];
            double std = Math.sqrt(2.0 / (channels * kernelSize * kernelSize));
            for (int c = 0; c < channels; c++)
                for (int i = 0; i < kernelSize; i++)
                    for (int j = 0; j < kernelSize; j++)
                        kernel[c][i][j] = rand.nextGaussian() * std;
            neurons[f] = new convNeuron(kernel, 0.0);
        }
        convLayer cL = new convLayer(neurons);

        activateLayer      aL = new activateLayer(new ReLU());      // ReLU：引入非线性
        poolingLayer       pL = new poolingLayer(2, 2);             // 2×2 最大池化
        flattenLayer       fL = new flattenLayer();                  // 展平为一维向量

        // ── 3. 动态推导 flatten 后的维度 ─────────────────────────────
        // 不手动计算 (64-5+1)/2 等尺寸，直接跑一次前向传播拿到真实维度
        loader.reset();
        double[][][] sample  = loader.next();
        double[]     flatOut = fL.forward(pL.forward(aL.forward(cL.forward(sample))));
        int flatSize = flatOut.length;
        System.out.println("flatten 后维度: " + flatSize);

        // ── 4. 初始化全连接层 ────────────────────────────────────────
        // He 初始化：权重 ~ N(0, sqrt(2 / flatSize))
        double[][] denseW = new double[numClasses][flatSize];
        double std = Math.sqrt(2.0 / flatSize);
        for (int i = 0; i < numClasses; i++)
            for (int j = 0; j < flatSize; j++)
                denseW[i][j] = rand.nextGaussian() * std;
        double[]            denseB = new double[numClasses];
        denseLayer          dL     = new denseLayer(denseW, denseB);
        activateVectorLayer avL    = new activateVectorLayer(new Sigmoid());

        CNN cnn = new CNN(cL, aL, pL, fL, dL, avL);

        // ── 5. 检测已保存模型，决定是加载还是重新训练 ─────────────────
        File modelFile = new File("model.txt");
        if (modelFile.exists()) {
            ModelIO.load(cL, dL, "model.txt");
            System.out.println("已加载保存的模型，跳过训练");
        } else {
            for (int epoch = 0; epoch < epochs; epoch++) {
                loader.reset();
                double totalLoss = 0;
                int idx = 0, correct = 0;

                while (loader.hasNext()) {
                    int label = loader.currentLabel();
                    double[][][] img = loader.next();
                    double[] out = cnn.forward(img);
                    double[] target = new double[numClasses];
                    target[label] = 1.0;
                    totalLoss += mseLoss(out, target);
                    double[] gradient = mseGradient(out, target);
                    cnn.backward(gradient, learningRate);
                    if (maxIndex(out) == label) correct++;
                    idx++;
                }

                System.out.printf("epoch=%d | avg loss=%.6f | accuracy=%d/%d%n",
                        epoch, totalLoss / idx, correct, idx);
            }

            ModelIO.save(cL, dL, "model.txt");
        }

        // ── 6. 逐张测试（加载或训练后均执行）────────────────────────────
        loader.reset();
        double totalLoss = 0;
        int idx = 0, correct = 0;
        while (loader.hasNext()) {
            String name  = loader.currentFileName();
            int    label = loader.currentLabel();
            double[][][] img = loader.next();
            double[] out    = cnn.forward(img);
            double[] target = new double[numClasses];
            target[label] = 1.0;
            totalLoss += mseLoss(out, target);
            int pred = maxIndex(out);
            if (pred == label) correct++;
            System.out.printf("[%d] %s | 真实=%d 预测=%d loss=%.4f%n", idx, name, label, pred, mseLoss(out, target));
            idx++;
        }
        System.out.printf("%n平均 loss: %.4f | accuracy=%d/%d%n", totalLoss / idx, correct, idx);

        // ── 7. 一致性测试：同一张图输两次，结果必须完全相同 ─────────────
        // 若结果不同，说明网络中存在随机性或状态未正确重置
        System.out.println("\n── 一致性测试 ──");
        loader.reset(); double[] out1 = cnn.forward(loader.next());
        loader.reset(); double[] out2 = cnn.forward(loader.next());
        boolean same = true;
        for (int i = 0; i < out1.length; i++) {
            if (Math.abs(out1[i] - out2[i]) > 1e-10) { same = false; break; }
        }
        System.out.println("两次预测结果一致: " + same);
    }

    // MSE = (1/2) * Σ(output[i] - target[i])²，系数1/2使导数形式简洁
    private static double mseLoss(double[] output, double[] target) {
        if (output.length != target.length) throw new IllegalArgumentException("length mismatch");
        double loss = 0;
        for (int i = 0; i < output.length; i++) {
            double d = output[i] - target[i];
            loss += d * d;
        }
        return loss / 2;
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

    // 返回概率最大的类别下标，作为预测结果
    private static int maxIndex(double[] values) {
        int max = 0;
        for (int i = 1; i < values.length; i++)
            if (values[i] > values[max]) max = i;
        return max;
    }
}
