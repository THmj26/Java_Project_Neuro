// 卷积层：管理多个卷积核（convNeuron），每个核独立提取一种特征，输出多张特征图
//   前向：N 个 filter 各自做一次卷积，输出 N 张特征图，堆叠成 [N][outRow][outCol]
//   反向：每个 filter 收到对应特征图的梯度，各自更新自身权重；
//         对输入的梯度 = 所有 filter 传回梯度的逐元素之和
public class convLayer {
    convNeuron[] filters; // 所有卷积核，每个 filter 独立提取一种特征

    convLayer(convNeuron[] c) {
        filters = c;
    }

    // ── 前向传播 ──────────────────────────────────────────────────────────────
    // input:  [channels][inputRow][inputCol]
    // 返回:   [numFilters][outRow][outCol]，每个 filter 输出一张特征图
    public double[][][] forward(double[][][] input) {
        double[][][] output = new double[filters.length][][]; // 输出数组，每个元素是一张特征图
        for (int i = 0; i < filters.length; i++) {           // 遍历每个 filter
            output[i] = filters[i].forward(input);           // 第 i 个 filter 独立做卷积，输出第 i 张特征图
        }
        return output; // 返回所有特征图堆叠的结果
    }

    // ── 反向传播 ──────────────────────────────────────────────────────────────
    // pdz:   [numFilters][outRow][outCol]，每张特征图各自对应的梯度
    // 返回:  [channels][inputRow][inputCol]，∂L/∂input，各 filter 梯度之和，传给 activateLayer
    public double[][][] backward(double[][][] pdz, double learningRate) {
        double[][][] gradient = null;                          // 累积所有 filter 对输入的梯度，初始为 null
        for (int i = 0; i < filters.length; i++) {            // 遍历每个 filter
            // 将第 i 张特征图的梯度传给第 i 个 filter，filter 内部更新自身 kernel，并返回对输入的梯度
            double[][][] oneNeuronGradient = filters[i].backward(pdz[i], learningRate);

            if (gradient == null) {
                // 第一个 filter 返回时，用其形状初始化累积梯度数组（全0）
                gradient = new double[oneNeuronGradient.length][oneNeuronGradient[0].length][oneNeuronGradient[0][0].length];
            }

            // 将当前 filter 对输入的梯度累加到总梯度中
            // 原因：同一输入像素被所有 filter 同时使用，∂L/∂input = Σ_filter ∂L/∂input_from_filter_i
            for (int p = 0; p < oneNeuronGradient.length; p++) {        // 遍历通道
                for (int k = 0; k < oneNeuronGradient[0].length; k++) { // 遍历行
                    for (int j = 0; j < oneNeuronGradient[0][0].length; j++) { // 遍历列
                        gradient[p][k][j] += oneNeuronGradient[p][k][j]; // 累加当前 filter 的梯度到总梯度
                    }
                }
            }
        }
        return gradient; // 返回汇总后的 ∂L/∂input，传给 activateLayer
    }
}
