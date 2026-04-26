public class convolutionLayer {
    double[][] kernel;
    double bias;
    convolutionLayer(double[][] k,double b){
        kernel=k;
        bias=b;
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
                        sum+=input[i+ki][j+kj]*kernel[kj][ki];
                        //对每一个运算后的矩阵上的所有点求和
                    }
                }
                output[i][j]=sum+bias;
            }
        }
        return output;
    }
}
