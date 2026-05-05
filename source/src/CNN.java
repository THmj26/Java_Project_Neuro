// CNN 主体：将所有层组装成完整的前向/反向传播管道
// 网络结构：Conv → ReLU → MaxPool → Flatten → Dense → Sigmoid
// 核心原理：
//   前向：输入依次流过每一层，最终输出 12 维概率向量
//   反向：从输出端的 MSE 梯度出发，按前向的逆序逐层传播梯度并更新权重（链式法则）
public class CNN {
    private convLayer           cL;  // 卷积层：提取局部特征
    private activateLayer       aL;  // ReLU 激活层：引入非线性，特征图逐元素激活
    private poolingLayer        pL;  // 最大池化层：空间下采样，压缩特征图
    private flattenLayer        fL;  // 展平层：三维特征图→一维向量
    private denseLayer          dL;  // 全连接层：特征向量→类别得分
    private activateVectorLayer avL; // Sigmoid 激活层：类别得分→(0,1)概率

    CNN(convLayer cL, activateLayer aL, poolingLayer pL,
        flattenLayer fL, denseLayer dL, activateVectorLayer avl) {
        this.cL  = cL;
        this.aL  = aL;
        this.pL  = pL;
        this.fL  = fL;
        this.dL  = dL;
        this.avL = avl;
    }

    // ── 前向传播 ──────────────────────────────────────────────────────────────
    // input: [1][64][64]（灰度单通道图片）
    // 返回:  [12]，每个值在 (0,1) 之间，表示对应生肖类别的预测概率
    public double[] forward(double[][][] input) {
        double[][][] convOutput    = cL.forward(input);        // 卷积：提取 8 张局部特征图 [8][60][60]
        double[][][] activatedConv = aL.forward(convOutput);   // ReLU：负值截0，保留正特征   [8][60][60]
        double[][][] pooled        = pL.forward(activatedConv); // 最大池化：尺寸减半          [8][30][30]
        double[]     flat          = fL.forward(pooled);       // 展平：三维→一维向量          [7200]
        double[]     denseOutput   = dL.forward(flat);         // 全连接：映射到12个类别得分   [12]
        return avL.forward(denseOutput);                       // Sigmoid：得分→(0,1)概率      [12]
    }

    // ── 反向传播 ──────────────────────────────────────────────────────────────
    // pdz:   ∂L/∂output，[12]，由 train.java 中 mseGradient 计算得到，是反向传播的起点
    // 反向顺序与前向完全相反：Sigmoid → Dense → Flatten → MaxPool → ReLU → Conv
    public void backward(double[] pdz, double learningRate) {
        double[]     g1 = avL.backward(pdz);              // Sigmoid 层：∂L/∂denseOutput = pdz * σ'(z)
        double[]     g2 = dL.backward(g1, learningRate);  // Dense 层：更新 weight/bias，返回 ∂L/∂flat
        double[][][] g3 = fL.backward(g2);                // Flatten 层：reshape 一维梯度→三维，值不变
        double[][][] g4 = pL.backward(g3);                // MaxPool 层：梯度路由到 max 位置，其余为0
        double[][][] g5 = aL.backward(g4);                // ReLU 层：梯度乘 ReLU'(z)，负值位置截断为0
        cL.backward(g5, learningRate);                    // Conv 层：更新所有 kernel/bias
    }
}
