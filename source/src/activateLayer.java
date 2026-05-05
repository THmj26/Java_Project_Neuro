// 特征图激活层：对卷积输出的每张特征图逐元素应用激活函数
// 核心原理：
//   前向：output[c][i][j] = f(input[c][i][j])，f 由构造时传入的激活函数决定
//   反向：链式法则，dL/dinput[c][i][j] = dL/doutput[c][i][j] * f'(input[c][i][j])
//         f' 需要激活前的原始值，因此必须缓存 lastInput
public class activateLayer {
    private activateFuction af;
    private double[][][] lastInput; // 缓存激活前的值，反向传播求导时使用

    activateLayer(activateFuction a) {
        af = a;
    }

    // 前向传播：对所有特征图逐元素激活
    // input/output: [numFilters][rows][cols]
    public double[][][] forward(double[][][] input) {
        lastInput = input.clone();
        int C = input.length;
        int R = input[0].length;
        int cols = input[0][0].length;

        double[][][] output = new double[C][R][cols];
        for (int c = 0; c < C; c++) {
            for (int i = 0; i < R; i++) {
                for (int j = 0; j < cols; j++) {
                    output[c][i][j] = af.activate(input[c][i][j]);
                }
            }
        }
        return output;
    }

    // 反向传播：将后层梯度乘以激活函数导数，传给卷积层
    // pdx: [numFilters][rows][cols]，即 ∂L/∂output
    // 返回: ∂L/∂input，形状相同
    public double[][][] backward(double[][][] pdx) {
        int C = pdx.length;
        int R = pdx[0].length;
        int cols = pdx[0][0].length;
        double[][][] gradient = new double[C][R][cols];
        for (int p = 0; p < C; p++) {
            for (int i = 0; i < R; i++) {
                for (int j = 0; j < cols; j++) {
                    gradient[p][i][j] = af.derivative(lastInput[p][i][j]) * pdx[p][i][j];
                }
            }
        }
        return gradient;
    }
}
