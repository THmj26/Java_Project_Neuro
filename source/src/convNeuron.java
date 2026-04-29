public class convNeuron {
    //一个卷积计算就是相当于之前NN中的一个神经元 是卷积运算中的基本单位
    double[][][] kernel;
    double bias;
    double[][][] lastInput;

    convNeuron(double[][][] k, double b){
        kernel=k;
        bias=b;
    }
    //这里是对传入的多个二维图像做卷积运算
    //每一张图像有其对应的kernel
    //每一个点的卷积结果等于 每一个kernel和对应输入 在这个点的卷积结果之和
    public double[][] forward(double[][][] input){
        lastInput = input;
        int pictures=input.length;
        int inputRow=input[0].length;
        int inputColumn=input[0][0].length;

        int kernelI=kernel.length;
        int kernelR=kernel[0].length;
        int kernelC=kernel[0][0].length;

        //这里的加一对应的是kernel这个矩阵和初始数据的第一个结果 差值是能走多少步 就会产生多少阶
        int outR=inputRow-kernelR+1;
        int outC=inputColumn-kernelC+1;

        double[][] output=new double[outR][outC];
        for(int i=0;i<outR;i++){
            for(int j=0;j<outC;j++){
                double sum=0;
                //外层循环用来给最终输出赋值
                //以及遍历输入的数据
                for (int c=0;c<pictures;c++){
                    //c可以用来控制第几张图片
                    //在这里可以讲kernel看作NN中每一个神经元的权重！
                    for(int ki=0;ki<kernelR;ki++){
                        for(int kj=0;kj<kernelC;kj++){
                            sum+=input[c][i+ki][j+kj]*kernel[c][ki][kj];
                        }
                    }
                }
                output[i][j]=sum+bias;
            }
        }
        return output;
    }
    public double[][][] backward(double[][] pdz, double learningRate) {
        int pictures = lastInput.length;
        int inputRow = lastInput[0].length;
        int inputColumn = lastInput[0][0].length;

        int kernelR = kernel[0].length;
        int kernelC = kernel[0][0].length;

        int outR = pdz.length;
        int outC = pdz[0].length;

        double[][][] pdx = new double[pictures][inputRow][inputColumn];
        double[][][] gradKernel = new double[pictures][kernelR][kernelC];
        double gradBias = 0.0;

        for (int i = 0; i < outR; i++) {
            for (int j = 0; j < outC; j++) {
                double grad = pdz[i][j];
                gradBias += grad;

                for (int c = 0; c < pictures; c++) {
                    for (int ki = 0; ki < kernelR; ki++) {
                        for (int kj = 0; kj < kernelC; kj++) {
                            int inputR = i + ki;
                            int inputC = j + kj;

                            gradKernel[c][ki][kj] += grad * lastInput[c][inputR][inputC];
                            pdx[c][inputR][inputC] += grad * kernel[c][ki][kj];
                            //这里求和的原因是因为 卷积中 一个输出会和多个k与x有关
                            //其实就是找最后结果 是对应原本输入中的那些数据
                            //所以想要寻找位置只需要再重新走一遍来时路就可以了
                            //那么从节点i，j位置来说kernel中每一个k对应的输入就是ki+i，kj+j

                            //同样的对于x的偏导来说也是一样的 求和x有关的k的和
                            //那么
                        }
                    }
                }
            }
        }

        for (int c = 0; c < pictures; c++) {
            for (int ki = 0; ki < kernelR; ki++) {
                for (int kj = 0; kj < kernelC; kj++) {
                    kernel[c][ki][kj] -= learningRate * gradKernel[c][ki][kj];
                }
            }
        }

        bias -= learningRate * gradBias;//这里的gradbias就是上一层中所有的梯度之和

        return pdx;
    }
}
