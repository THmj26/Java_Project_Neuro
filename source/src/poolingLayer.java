// 最大池化层：对特征图做下采样，压缩空间尺寸，保留区域内最显著的特征
// 核心原理：
//   前向：将特征图划分为 poolSize×poolSize 的窗口，每个窗口取最大值输出
//         outSize = (inputSize - poolSize) / stride + 1
//   反向：梯度只流向前向传播时取到最大值的那个位置（max位置路由）；
//         其他位置梯度为0，因为它们对输出没有贡献
//         实现方式：反向时重走一遍前向逻辑，定位每个输出对应的 max 位置，将梯度写入该位置
public class poolingLayer {
    private int poolSize;
    private int stride;
    private double[][][] lastInput; // 缓存前向输入，反向时重新定位 max 位置

    poolingLayer(int size, int stride) {
        poolSize = size;
        this.stride = stride;
    }

    // ── 前向传播 ──────────────────────────────────────────────────────────────
    // input: [numFilters][inputRow][inputCol]
    // 返回:  [numFilters][outRow][outCol]，每个窗口取最大值
    public double[][][] forward(double[][][] input) {
        lastInput = new double[input.length][input[0].length][input[0][0].length];
        for (int c = 0; c < input.length; c++)
            for (int i = 0; i < input[c].length; i++)
                lastInput[c][i] = input[c][i].clone(); // 深拷贝，防止反向传播时原数组被修改
        int C    = input.length;                // 特征图数量
        int inR  = input[0].length;             // 输入特征图行数
        int inC  = input[0][0].length;          // 输入特征图列数
        int outR = (inR - poolSize) / stride + 1; // 输出行数
        int outC = (inC - poolSize) / stride + 1; // 输出列数

        double[][][] output = new double[C][outR][outC]; // 输出特征图，初始全0
        for (int c = 0; c < C; c++) {                    // 遍历每张特征图
            for (int i = 0; i < outR; i++) {             // 遍历输出的每一行
                for (int j = 0; j < outC; j++) {         // 遍历输出的每一列
                    double max = Double.NEGATIVE_INFINITY; // 当前窗口的最大值，初始为负无穷
                    for (int ki = 0; ki < poolSize; ki++) {   // 遍历池化窗口的每一行
                        for (int kj = 0; kj < poolSize; kj++) { // 遍历池化窗口的每一列
                            // 窗口内输入位置 = 输出位置 * stride + 窗口内偏移
                            max = Math.max(max, input[c][i * stride + ki][j * stride + kj]);
                        }
                    }
                    output[c][i][j] = max; // 该窗口的输出 = 窗口内最大值
                }
            }
        }
        return output;
    }

    // ── 反向传播 ──────────────────────────────────────────────────────────────
    // pdx:   ∂L/∂output，[numFilters][outRow][outCol]，来自 flattenLayer 传回的梯度
    // 返回:  ∂L/∂input，[numFilters][inputRow][inputCol]，传给 activateLayer
    public double[][][] backward(double[][][] pdx) {
        if (lastInput == null) throw new IllegalArgumentException("never called forward!");
        int C    = lastInput.length;                          // 特征图数量
        int inR  = lastInput[0].length;                       // 输入行数
        int inC  = lastInput[0][0].length;                    // 输入列数
        int outR = (inR - poolSize) / stride + 1;             // 输出行数（与前向一致）
        int outC = (inC - poolSize) / stride + 1;             // 输出列数

        double[][][] gradient = new double[C][inR][inC]; // ∂L/∂input，初始全0（非 max 位置梯度为0）

        for (int c = 0; c < C; c++) {                    // 遍历每张特征图
            for (int i = 0; i < outR; i++) {             // 遍历输出的每一行
                for (int j = 0; j < outC; j++) {         // 遍历输出的每一列
                    double max = Double.NEGATIVE_INFINITY; // 重新寻找该窗口的最大值
                    int maxR = i * stride, maxC = j * stride; // 初始化为窗口左上角，确保始终有效
                    for (int ki = 0; ki < poolSize; ki++) {   // 遍历池化窗口的每一行
                        for (int kj = 0; kj < poolSize; kj++) { // 遍历池化窗口的每一列
                            int r   = i * stride + ki;     // 对应输入的行索引
                            int col = j * stride + kj;     // 对应输入的列索引
                            if (lastInput[c][r][col] > max) { // 找到新的最大值
                                max  = lastInput[c][r][col];  // 更新最大值
                                maxR = r;                     // 记录最大值所在行
                                maxC = col;                   // 记录最大值所在列
                            }
                        }
                    }
                    // 梯度只传给前向时取到 max 的位置，其余位置保持0（max 以外的元素对输出无贡献）
                    gradient[c][maxR][maxC] = pdx[c][i][j];
                }
            }
        }
        return gradient; // 传给 activateLayer 继续反向传播
    }
}
