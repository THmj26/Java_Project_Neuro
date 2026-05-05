// MSE（均方误差）损失函数：衡量预测值与真实标签的差距
// 核心原理：
//   损失：L = (1/2) * Σ (predict[i] - target[i])²
//         系数 1/2 是为了让求导后形式简洁（导数恰好等于误差本身，无多余系数）
//   梯度：∂L/∂predict[i] = predict[i] - target[i]
//         梯度方向指向预测值偏离目标值的方向，SGD 沿负梯度方向更新以减小误差
public class MseLoss {

    // 计算 MSE 损失，系数 1/2 使导数形式简洁
    public double loss(double[] target, double[] predict) {
        double sum = 0;
        for (int i = 0; i < predict.length; i++) {
            sum += Math.pow(target[i] - predict[i], 2);
        }
        return sum / 2;
    }

    // 计算损失对预测值的梯度：∂L/∂predict[i] = predict[i] - target[i]
    public double[] gradient(double[] target, double[] predict) {
        double[] g = new double[predict.length];
        for (int i = 0; i < predict.length; i++) {
            g[i] = predict[i] - target[i];
        }
        return g;
    }
}
