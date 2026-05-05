// 全连接层（线性变换层）：将展平后的向量映射到输出类别空间
// 核心原理：
//   前向：output[i] = Σ_j weight[i][j] * input[j] + bias[i]，即矩阵乘法 + 偏置
//   反向：需要求三个梯度
//         ∂L/∂weight[i][j] = pdz[i] * input[j]           （用于更新权重）
//         ∂L/∂bias[i]      = pdz[i]                       （用于更新偏置）
//         ∂L/∂input[j]     = Σ_i pdz[i] * weight[i][j]   （传给展平层，转置矩阵乘法）
public class denseLayer {
    double[][] weight;  // [outputSize][inputSize]，每行是一个神经元的权重向量
    double[]   bias;    // [outputSize]
    double[]   lastInput; // 缓存前向输入，反向计算 ∂L/∂weight 时使用

    public denseLayer(double[][] w, double[] b) {
        weight = w;
        bias = b;
    }

    // ── 前向传播 ──────────────────────────────────────────────────────────────
    // input:  [inputSize]，展平层输出的一维向量
    // 返回:   [outputSize]，线性变换结果（激活前）
    public double[] forward(double[] input) {
        int outputSize = weight.length;     // 输出维度 = 神经元数量（本项目为12，对应12生肖）
        int inputSize  = input.length;      // 输入维度 = flatten 后的特征向量长度
        lastInput = input.clone();          // 缓存输入，反向计算 ∂L/∂weight 时需要

        double[] output = new double[outputSize]; // 输出向量，初始全0
        for (int i = 0; i < outputSize; i++) {    // 遍历每个输出神经元
            double sum = 0;                       // 第 i 个神经元的加权求和
            for (int j = 0; j < inputSize; j++) { // 遍历每个输入特征
                sum += weight[i][j] * input[j];   // 加权：weight[i][j] 是第 i 个神经元对第 j 个输入的权重
            }
            output[i] = sum + bias[i];            // 加上偏置，得到第 i 个神经元的输出
        }
        return output;
    }

    // ── 反向传播 ──────────────────────────────────────────────────────────────
    // pdz:   ∂L/∂output，[outputSize]，来自激活层（activateVectorLayer）传回的梯度
    // 返回:  pdx = ∂L/∂input，[inputSize]，传给展平层
    double[] backward(double[] pdz, double learningRate) {
        if (lastInput == null) throw new IllegalArgumentException("never called forward!");
        if (pdz.length != weight.length)  throw new IllegalArgumentException("pdz's length must match neurons' size");

        double[] pdx = new double[weight[0].length]; // ∂L/∂input，初始全0
        for (int i = 0; i < weight.length; i++) {    // 遍历每个输出神经元
            for (int j = 0; j < weight[0].length; j++) { // 遍历每个输入特征
                // ∂L/∂input[j] = Σ_i pdz[i] * weight[i][j]，转置矩阵乘法，需累加所有神经元的贡献
                pdx[j] += pdz[i] * weight[i][j];
                // ∂L/∂weight[i][j] = pdz[i] * input[j]，再做 SGD 更新
                weight[i][j] -= pdz[i] * lastInput[j] * learningRate;
            }
            bias[i] -= pdz[i] * learningRate; // ∂L/∂bias[i] = pdz[i]，SGD 更新偏置
        }
        return pdx; // 传给展平层继续反向传播
    }
}
