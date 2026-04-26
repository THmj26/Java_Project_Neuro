public class convNeuron {
    //一个卷积计算就是相当于之前NN中的一个神经元 是卷积运算中的基本单位
    double[][][] kernel;
    double bias;
    convNeuron(double[][][] k, double b){
        kernel=k;
        bias=b;
    }
    //这里是对传入的多个二维图像做卷积运算
    //每一张图像有其对应的kernel
    //每一个点的卷积结果等于 每一个kernel和对应输入 在这个点的卷积结果之和
    public double[][] forward(double[][][] input){
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
                    for(int ki=0;ki<kernelR;ki++){
                        for(int kj=0;kj<kernelC;kj++){
                            sum+=input[c][i+ki][j+kj]*kernel[c][ki][kj];
                        }
                    }
                }
                output[i][j]=sum+bias;
            }
        }
//        最外层的ij代表着起始点
//        for(int i=0;i<outR;i++){
//            for(int j=0;j<outC;j++){
//                double sum=0.0;
//                for(int ki=0;ki<kernelR;ki++){
//                    for(int kj=0;kj<kernelC;kj++){
//                        sum+=input[i+ki][j+kj]*kernel[ki][kj];
//                        //卷积运算和矩阵乘法不同 其是矩阵对应位置相乘
//                        //对每一个运算后的矩阵上的所有点求和
//                    }
//                }
//                output[i][j]=sum+bias;
//            }
//        }
        return output;
    }
}
