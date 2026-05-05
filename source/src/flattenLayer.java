// 展平层：将三维特征图拉成一维向量，连接卷积部分与全连接部分
// 核心原理：
//   前向：按 [通道][行][列] 顺序将所有元素依次写入一维数组，不做任何数值计算
//   反向：纯粹的形状变换，将全连接层传回的一维梯度按同样顺序恢复为三维，梯度值不变
public class flattenLayer {
    int lastI; // 缓存前向输入的通道数，反向时用于恢复三维形状
    int lastR; // 缓存前向输入的行数
    int lastC; // 缓存前向输入的列数

    // ── 前向传播 ──────────────────────────────────────────────────────────────
    // input:  [numFilters][rows][cols]
    // 返回:   [numFilters * rows * cols]，一维向量
    public double[] forward(double[][][] input) {
        lastI = input.length;           // 记录通道数，反向时恢复形状使用
        lastR = input[0].length;        // 记录行数
        lastC = input[0][0].length;     // 记录列数

        double[] output = new double[lastI * lastR * lastC]; // 一维输出数组
        int index = 0;                  // 当前写入位置的指针
        for (double[][] channel : input) {      // 遍历每个通道
            for (int i = 0; i < lastR; i++) {   // 遍历每一行
                for (int j = 0; j < lastC; j++) { // 遍历每一列
                    output[index++] = channel[i][j]; // 按顺序写入，index 自增
                }
            }
        }
        return output;
    }

    // ── 反向传播 ──────────────────────────────────────────────────────────────
    // pdx:   ∂L/∂output，[numFilters * rows * cols]，来自全连接层传回的梯度
    // 返回:  ∂L/∂input，[numFilters][rows][cols]，传给池化层（形状恢复，梯度值不变）
    public double[][][] backward(double[] pdx) {
        if (pdx.length != lastC * lastR * lastI)
            throw new IllegalArgumentException("pdx's length doesn't match flatten input/output size");

        double[][][] gradient = new double[lastI][lastR][lastC]; // 三维梯度数组
        int index = 0;                  // 读取一维梯度的指针，顺序与 forward 展开完全一致
        for (int c = 0; c < lastI; c++) {       // 遍历每个通道
            for (int i = 0; i < lastR; i++) {   // 遍历每一行
                for (int j = 0; j < lastC; j++) { // 遍历每一列
                    gradient[c][i][j] = pdx[index++]; // 按 forward 展开的相同顺序填回，梯度值不变
                }
            }
        }
        return gradient; // 传给池化层继续反向传播
    }
}
