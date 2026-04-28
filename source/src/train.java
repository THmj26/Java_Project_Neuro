import java.io.IOException;
import java.util.Random;

// 文件：train.java
// 功能：构建CNN网络，加载十二生肖图片，执行前向传播并计算MSE Loss
// 当前阶段：仅前向传播，权重随机初始化，尚未实现反向传播训练
public class train {
    public static void main(String[] args) throws IOException {

        // ══════════════════════════════════════════════════════════
        // 超参数配置
        // ══════════════════════════════════════════════════════════
        String imageDir = "images";  // 图片目录（64x64灰度图）
        int numClasses  = 12;        // 输出类别数：十二生肖
        int numFilters  = 8;         // 卷积层filter数量
        int kernelSize  = 5;         // 卷积核大小：5x5
        int channels    = 1;         // 输入通道数：灰度图为1

        // ══════════════════════════════════════════════════════════
        // 1. 加载图片
        // ══════════════════════════════════════════════════════════
        DataLoader loader = new DataLoader(imageDir, true);
        System.out.println("图片总数: " + loader.size());

        // ══════════════════════════════════════════════════════════
        // 2. 初始化卷积层
        // ══════════════════════════════════════════════════════════
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

        // ReLU激活层：将负值置0，引入非线性
        activateLayer aL = new activateLayer(new ReLU());

        // 最大池化层(2x2)：图像尺寸减半，压缩特征图
        poolingLayer pL  = new poolingLayer(2, 2);

        // 展平层：三维特征图拉成一维向量
        flattenLayer fL  = new flattenLayer();

        // ══════════════════════════════════════════════════════════
        // 3. 动态计算flatten后的维度
        // ══════════════════════════════════════════════════════════
        loader.reset();
        double[][][] sample  = loader.next();
        double[][][] convOut = cL.forward(sample);
        double[][][] reluOut = aL.forward(convOut);
        double[][][] poolOut = pL.forward(reluOut);
        double[]     flatOut = fL.forward(poolOut);
        int flatSize = flatOut.length;
        System.out.println("flatten 后维度: " + flatSize);

        // ══════════════════════════════════════════════════════════
        // 4. 初始化全连接层
        // ══════════════════════════════════════════════════════════
        double[][] denseW = new double[numClasses][flatSize];
        double std = Math.sqrt(2.0 / flatSize);
        for (int i = 0; i < numClasses; i++) {
            for (int j = 0; j < flatSize; j++) {
                denseW[i][j] = rand.nextGaussian() * std;
            }
        }
        double[] denseB = new double[numClasses];
        denseLayer dL = new denseLayer(denseW, denseB);

        // Sigmoid激活：输出压缩到(0,1)
        activateVectorLayer avL = new activateVectorLayer(new Sigmoid());

        // 组装CNN：卷积→激活→池化→展平→全连接→激活
        CNN cnn = new CNN(cL, aL, pL, fL, dL, avL);

        // ══════════════════════════════════════════════════════════
        // 5. 前向传播 + 计算MSE Loss
        // ══════════════════════════════════════════════════════════
        loader.reset();
        double totalLoss = 0;
        int idx = 0;

        while (loader.hasNext()) {
            String name = loader.currentFileName();
            double[][][] img = loader.next();

            // 前向传播：图片 → 12个生肖的预测概率
            double[] out = cnn.forward(img);

            // one-hot目标向量：第idx个位置为1，其余为0
            double[] target = new double[numClasses];
            target[idx] = 1.0;

            // MSE Loss = mean((output - target)^2)
            double loss = 0;
            for (int i = 0; i < numClasses; i++) {
                double diff = out[i] - target[i];
                loss += diff * diff;
            }
            loss /= numClasses;
            totalLoss += loss;

            // 找预测概率最高的类别
            int pred = 0;
            for (int i = 1; i < numClasses; i++) {
                if (out[i] > out[pred]) pred = i;
            }

            System.out.printf("[%d] %s | 真实=%d 预测=%d loss=%.4f%n",
                    idx, name, idx, pred, loss);
            idx++;
        }

        System.out.printf("%n平均 loss: %.4f%n", totalLoss / idx);

        // ══════════════════════════════════════════════════════════
        // 6. 一致性测试：同一张图片输入两次，预测结果应完全相同
        // ══════════════════════════════════════════════════════════
        System.out.println("\n── 一致性测试 ──");
        loader.reset();
        double[][][] img1 = loader.next();
        double[] out1 = cnn.forward(img1);

        loader.reset();
        double[][][] img2 = loader.next();
        double[] out2 = cnn.forward(img2);

        // 比较两次输出是否完全一致
        boolean same = true;
        for (int i = 0; i < out1.length; i++) {
            if (Math.abs(out1[i] - out2[i]) > 1e-10) {
                same = false;
                break;
            }
        }
        System.out.println("两次预测结果一致: " + same);
        System.out.print("第一次输出: ");
        for (double v : out1) System.out.printf("%.4f ", v);
        System.out.print("\n第二次输出: ");
        for (double v : out2) System.out.printf("%.4f ", v);
        System.out.println();
    }
}
