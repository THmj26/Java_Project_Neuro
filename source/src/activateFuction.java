// 激活函数接口：所有激活函数必须实现正向激活和导数两个方法
// 核心原理：激活函数引入非线性，使网络能拟合非线性映射；
//           导数用于反向传播中的链式求导 dL/dx = dL/dy * f'(x)
public interface activateFuction {
    // 正向激活：给定输入 x，返回激活值 f(x)
    double activate(double x);

    // 求导：给定输入 z，返回激活函数在该点的导数 f'(z)
    // 注意：传入的是激活前的原始值 z，不是激活后的输出
    double derivative(double z);
}
