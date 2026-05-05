// 单个卷积核（卷积神经元）：CNN 中最基本的计算单元，类比全连接层中的单个神经元
// 核心原理：
//   前向：对输入的每个通道与对应 kernel 做滑动互相关，所有通道结果相加得到一张输出特征图
//         output[i][j] = Σ_c Σ_ki Σ_kj  input[c][i+ki][j+kj] * kernel[c][ki][kj]  + bias
//   反向：需要求三个梯度
//         ∂L/∂kernel[c][ki][kj] = Σ_i Σ_j  pdz[i][j] * input[c][i+ki][j+kj]   （用于更新权重）
//         ∂L/∂input[c][r][s]    = Σ_i Σ_j  pdz[i][j] * kernel[c][r-i][s-j]     （传给上一层）
//         ∂L/∂bias              = Σ_i Σ_j  pdz[i][j]                             （更新偏置）
public class convNeuron {
    double[][][] kernel;    // [通道数][核行][核列]
    double bias;
    double[][][] lastInput; // 缓存前向输入，反向传播时使用

    convNeuron(double[][][] k, double b) {
        kernel = k;
        bias = b;
    }

    // ── 前向传播 ──────────────────────────────────────────────────────────────
    // input: [channels][inputRow][inputCol]
    // 返回:  [outRow][outCol]，outRow = inputRow - kernelR + 1（无 padding，每步滑动1格）
    public double[][] forward(double[][][] input) {
        lastInput = new double[input.length][input[0].length][input[0][0].length];
        for (int c = 0; c < input.length; c++)
            for (int i = 0; i < input[c].length; i++)
                lastInput[c][i] = input[c][i].clone(); // 深拷贝，防止外部修改污染反向传播
        int pictures    = input.length;         // 输入通道数（灰度图为1）
        int inputRow    = input[0].length;      // 输入特征图的行数
        int inputColumn = input[0][0].length;   // 输入特征图的列数
        int kernelR     = kernel[0].length;     // 卷积核的行数
        int kernelC     = kernel[0][0].length;  // 卷积核的列数

        int outR = inputRow    - kernelR + 1;   // 输出行数：滑动窗口能放下的次数
        int outC = inputColumn - kernelC + 1;   // 输出列数：同上

        double[][] output = new double[outR][outC]; // 输出特征图，初始全0
        for (int i = 0; i < outR; i++) {            // 遍历输出特征图的每一行
            for (int j = 0; j < outC; j++) {        // 遍历输出特征图的每一列
                double sum = 0;                     // 当前输出位置的累加值
                for (int c = 0; c < pictures; c++) {        // 遍历每个输入通道
                    for (int ki = 0; ki < kernelR; ki++) {  // 遍历核的每一行
                        for (int kj = 0; kj < kernelC; kj++) { // 遍历核的每一列
                            // 输入位置 = 输出位置(i,j) + 核内偏移(ki,kj)
                            // 对应元素相乘后累加，即互相关运算
                            sum += input[c][i + ki][j + kj] * kernel[c][ki][kj];
                        }
                    }
                }
                output[i][j] = sum + bias; // 加上偏置得到最终输出值
            }
        }
        return output;
    }

    // ── 反向传播 ──────────────────────────────────────────────────────────────
    // pdz:   ∂L/∂output，[outRow][outC]，来自后一层（激活层）传回的梯度
    // 返回:  pdx = ∂L/∂input，[channels][inputRow][inputCol]，继续向前传给上一层
    public double[][][] backward(double[][] pdz, double learningRate) {
        int pictures    = lastInput.length;         // 通道数
        int inputRow    = lastInput[0].length;      // 输入行数
        int inputColumn = lastInput[0][0].length;   // 输入列数
        int kernelR     = kernel[0].length;         // 核行数
        int kernelC     = kernel[0][0].length;      // 核列数
        int outR        = pdz.length;               // 输出梯度行数
        int outC        = pdz[0].length;            // 输出梯度列数

        double[][][] pdx        = new double[pictures][inputRow][inputColumn]; // ∂L/∂input，初始全0
        double[][][] gradKernel = new double[pictures][kernelR][kernelC];      // ∂L/∂kernel，初始全0
        double gradBias = 0.0;                      // ∂L/∂bias，初始为0

        for (int i = 0; i < outR; i++) {            // 遍历输出梯度的每一行
            for (int j = 0; j < outC; j++) {        // 遍历输出梯度的每一列
                double grad = pdz[i][j];            // 当前输出位置的梯度值
                gradBias += grad;                   // ∂L/∂bias = Σ 所有输出位置的梯度之和

                for (int c = 0; c < pictures; c++) {        // 遍历每个通道
                    for (int ki = 0; ki < kernelR; ki++) {  // 遍历核的每一行
                        for (int kj = 0; kj < kernelC; kj++) { // 遍历核的每一列
                            // ∂L/∂kernel[c][ki][kj]：产生 output[i][j] 时用到了 input[c][i+ki][j+kj]
                            // 所以链式法则：∂L/∂kernel += ∂L/∂output * ∂output/∂kernel = grad * input
                            gradKernel[c][ki][kj] += grad * lastInput[c][i + ki][j + kj];

                            // ∂L/∂input[c][i+ki][j+kj]：该输入位置被多个输出位置共享，因此需要累加
                            // 链式法则：∂L/∂input += ∂L/∂output * ∂output/∂input = grad * kernel
                            pdx[c][i + ki][j + kj] += grad * kernel[c][ki][kj];
                        }
                    }
                }
            }
        }

        // SGD 权重更新：kernel -= lr * ∂L/∂kernel
        for (int c = 0; c < pictures; c++) {
            for (int ki = 0; ki < kernelR; ki++) {
                for (int kj = 0; kj < kernelC; kj++) {
                    kernel[c][ki][kj] -= learningRate * gradKernel[c][ki][kj];
                }
            }
        }
        bias -= learningRate * gradBias; // SGD 偏置更新：bias -= lr * ∂L/∂bias

        return pdx; // 将 ∂L/∂input 传给前一层（activateLayer）继续反向传播
    }
}
