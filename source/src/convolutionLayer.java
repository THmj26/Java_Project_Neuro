public class convolutionLayer {
    //一个卷积计算就是相当于之前NN中的一个神经元 是卷积运算中的基本单位
    double[][] kernel;
    double bias;
    private activateFuction af;
    convolutionLayer(double[][] k,double b,activateFuction a){
        kernel=k;
        bias=b;
        af=a;
    }
    //这里是对传入的二维数据做一个卷积运算
    public double[][] forward(double[][] input){
        int inputRow=input.length;
        int inputColumn=input[0].length;

        int kernelR=kernel.length;
        int kernelC=kernel[0].length;

        //这里的加一对应的是kernel这个矩阵和初始数据的第一个结果 差值是能走多少步 就会产生多少阶
        int outR=inputRow-kernelR+1;
        int outC=inputColumn-kernelC+1;

        double[][] output=new double[outR][outC];
        //最外层的ij代表着起始点
        for(int i=0;i<outR;i++){
            for(int j=0;j<outC;j++){
                double sum=0.0;
                for(int ki=0;ki<kernelR;ki++){
                    for(int kj=0;kj<kernelC;kj++){
                        sum+=input[i+ki][j+kj]*kernel[ki][kj];
                        //卷积运算和矩阵乘法不同 其是矩阵对应位置相乘
                        //对每一个运算后的矩阵上的所有点求和
                    }
                }
                output[i][j]=af.activate(sum+bias);
                //这里的af和之前在NN中的意义是一样的就是对输出结果进行激活
                //在卷积中的激活函数是对每一个矩阵中的元素进行激活
            }
        }
        return output;
    }
}
