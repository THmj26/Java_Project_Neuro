import java.io.*;
import java.util.Scanner;

// 模型权重的保存与加载
// 格式：纯文本，每行一个 double，按固定顺序写入/读取：
//   conv filter0 kernel → conv filter0 bias → filter1 ... → dense weight → dense bias
// 只保存权重数值，网络结构（层数、维度）由代码本身保证一致
public class ModelIO {

    // 将训练好的权重写入文件
    public static void save(convLayer cL, denseLayer dL, String path) throws IOException {
        PrintWriter pw = new PrintWriter(new FileWriter(path));

        // 逐个 filter 保存 kernel 所有值，再保存 bias
        for (convNeuron neuron : cL.filters) {
            for (double[][] channel : neuron.kernel)
                for (double[] row : channel)
                    for (double v : row)
                        pw.println(v);
            pw.println(neuron.bias);
        }

        // 保存全连接层权重矩阵（逐行）
        for (double[] row : dL.weight)
            for (double v : row)
                pw.println(v);

        // 保存全连接层偏置
        for (double v : dL.bias)
            pw.println(v);

        pw.close();
        System.out.println("模型已保存到: " + path);
    }

    // 从文件读取权重，填入已初始化的层对象（架构必须与保存时完全一致）
    public static void load(convLayer cL, denseLayer dL, String path) throws IOException {
        Scanner sc = new Scanner(new File(path));

        for (convNeuron neuron : cL.filters) {
            for (double[][] channel : neuron.kernel)
                for (int r = 0; r < channel.length; r++)
                    for (int c = 0; c < channel[r].length; c++)
                        channel[r][c] = sc.nextDouble();
            neuron.bias = sc.nextDouble();
        }

        for (int i = 0; i < dL.weight.length; i++)
            for (int j = 0; j < dL.weight[i].length; j++)
                dL.weight[i][j] = sc.nextDouble();

        for (int i = 0; i < dL.bias.length; i++)
            dL.bias[i] = sc.nextDouble();

        sc.close();
        System.out.println("模型已从 " + path + " 加载");
    }
}
