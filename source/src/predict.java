import java.io.IOException;
import java.util.Random;

// 推理脚本：加载保存的权重，对指定图片进行十二生肖分类
// 用法：java predict <图片路径>
//   例：java predict images/01_rat.jpg
//       java predict my_photo.jpg
public class predict {

    static final String[] CLASS_NAMES = {
        "Rat", "Ox", "Tiger", "Rabbit", "Dragon", "Snake",
        "Horse", "Goat", "Monkey", "Rooster", "Dog", "Pig"
    };

    public static void main(String[] args) throws IOException {
        if (args.length == 0) {
            System.out.println("用法: java predict <图片路径>");
            return;
        }
        String imagePath = args[0];

        // ── 1. 初始化与训练时完全相同的网络架构 ──────────────────────
        int numClasses  = 12;
        int numFilters  = 8;
        int kernelSize  = 5;
        int channels    = 1;

        Random rand = new Random(42); // 随机种子与 train.java 一致，保证 flatten 维度推导正确
        convNeuron[] neurons = new convNeuron[numFilters];
        for (int f = 0; f < numFilters; f++) {
            double[][][] kernel = new double[channels][kernelSize][kernelSize];
            neurons[f] = new convNeuron(kernel, 0.0);
        }
        convLayer           cL  = new convLayer(neurons);
        activateLayer       aL  = new activateLayer(new ReLU());
        poolingLayer        pL  = new poolingLayer(2, 2);
        flattenLayer        fL  = new flattenLayer();

        // 推导 flatSize（跑一张随机样本的前向，只为确定维度）
        double[][][] dummy   = new double[1][64][64];
        double[]     flatOut = fL.forward(pL.forward(aL.forward(cL.forward(dummy))));
        int flatSize = flatOut.length;

        double[][]          denseW = new double[numClasses][flatSize];
        double[]            denseB = new double[numClasses];
        denseLayer          dL     = new denseLayer(denseW, denseB);
        activateVectorLayer avL    = new activateVectorLayer(new Sigmoid());

        CNN cnn = new CNN(cL, aL, pL, fL, dL, avL);

        // ── 2. 加载训练好的权重 ───────────────────────────────────────
        ModelIO.load(cL, dL, "model.txt");

        // ── 3. 读取图片并推理 ─────────────────────────────────────────
        double[][][] img = loadSingleImage(imagePath);
        double[] output = cnn.forward(img);

        // ── 4. 输出结果 ───────────────────────────────────────────────
        int pred = maxIndex(output);
        System.out.printf("预测结果: %s（类别 %d）%n", CLASS_NAMES[pred], pred);
        System.out.println("\n各类别概率:");
        for (int i = 0; i < numClasses; i++) {
            System.out.printf("  [%2d] %-2s  %.4f%s%n",
                i, CLASS_NAMES[i], output[i], i == pred ? " ← 预测" : "");
        }
    }

    // 读取单张图片，复用 DataLoader 的灰度转换逻辑
    private static double[][][] loadSingleImage(String path) throws IOException {
        java.awt.image.BufferedImage img = javax.imageio.ImageIO.read(new java.io.File(path));
        if (img == null) throw new IOException("无法读取图片: " + path);
        int h = img.getHeight(), w = img.getWidth();
        double[][] channel = new double[h][w];
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                int rgb = img.getRGB(j, i);
                int r = (rgb >> 16) & 0xFF;
                int g = (rgb >> 8)  & 0xFF;
                int b =  rgb        & 0xFF;
                channel[i][j] = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0;
            }
        }
        return new double[][][]{channel};
    }

    private static int maxIndex(double[] values) {
        int max = 0;
        for (int i = 1; i < values.length; i++)
            if (values[i] > values[max]) max = i;
        return max;
    }
}
