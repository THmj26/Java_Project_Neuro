// ReLU（线性整流函数）激活函数
// 核心原理：f(x) = max(0, x)
//   正向：负值截断为0，正值直通，计算极快且缓解梯度消失
//   反向：导数是阶跃函数，x>0 时梯度为1（原样传递），x≤0 时梯度为0（截断传播）
public class ReLU implements activateFuction {

    // f(x) = max(0, x)
    public double activate(double x) {
        return Math.max(0, x);
    }

    // f'(z) = 1 if z>0, else 0
    public double derivative(double z) {
        return z > 0 ? 1 : 0;
    }
}
