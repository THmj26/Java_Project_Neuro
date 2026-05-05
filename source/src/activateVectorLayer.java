// 向量激活层：对一维向量（全连接层输出）逐元素应用激活函数
// 核心原理：与 activateLayer 相同，区别仅在于输入是 1D 而非 3D
//   前向：output[i] = f(input[i])
//   反向：dL/dinput[i] = dL/doutput[i] * f'(input[i])
public class activateVectorLayer {
    private activateFuction af;
    private double[] lastInput; // 缓存激活前的值，反向传播求导时使用

    activateVectorLayer(activateFuction a) {
        af = a;
    }

    // ── 前向传播 ──────────────────────────────────────────────────────────────
    // input:  [outputSize]，全连接层的线性输出（未激活）
    // 返回:   [outputSize]，逐元素激活后的概率向量
    public double[] forward(double[] input) {
        lastInput = input.clone();          // 缓存激活前原始值，反向传播计算 f'(z) 时需要
        double[] output = new double[input.length]; // 输出形状与输入相同
        for (int i = 0; i < input.length; i++) {    // 遍历向量每个元素
            output[i] = af.activate(input[i]);      // 逐元素激活：output[i] = f(input[i])
        }
        return output;
    }

    // ── 反向传播 ──────────────────────────────────────────────────────────────
    // pdy:   ∂L/∂output，[outputSize]，来自 MSE 梯度（反向传播的起点）
    // 返回:  ∂L/∂input，[outputSize]，传给全连接层
    public double[] backward(double[] pdy) {
        double[] g = new double[pdy.length]; // ∂L/∂input，初始全0
        for (int i = 0; i < pdy.length; i++) { // 遍历每个元素
            // 链式法则：∂L/∂input[i] = ∂L/∂output[i] * f'(激活前原始值)
            g[i] = af.derivative(lastInput[i]) * pdy[i];
        }
        return g; // 传给全连接层继续反向传播
    }
}
