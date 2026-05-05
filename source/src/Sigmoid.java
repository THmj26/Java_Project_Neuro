// Sigmoid 激活函数
// 核心原理：f(x) = 1 / (1 + e^(-x))，将任意实数压缩到 (0, 1)
//   正向：输出可解释为概率
//   反向：导数有简洁形式 f'(z) = f(z) * (1 - f(z))，避免重复计算
//   注意：深层网络中 Sigmoid 易导致梯度消失（导数最大值仅0.25），卷积层通常用 ReLU 代替
public class Sigmoid implements activateFuction {

    // f(x) = 1 / (1 + e^(-x))
    public double activate(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    // f'(z) = f(z) * (1 - f(z))
    public double derivative(double z) {
        double s = activate(z);
        return s * (1 - s);
    }
}
