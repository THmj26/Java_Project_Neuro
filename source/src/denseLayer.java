//这一层的作用是对flat层中的数据进行加权激活处理
//同样也是一组输入喂给多个神经元
public class denseLayer {
    double[][] weight;
    //二阶数组 一行代表一个神经元的权值
    double[] bias;
    double[] lastInput;

    public denseLayer(double[][] w,double[] b){
        weight=w;
        bias=b;
    }
    public double[] forward(double[] input){
        int outputSize=weight.length;
        int inputSize=input.length;
        lastInput=input.clone();

        double[] output=new double[outputSize];
        for(int i=0;i<outputSize;i++){
            double sum=0;
            for(int j=0;j<inputSize;j++){
                sum+=weight[i][j]*input[j];
            }
            output[i]=sum+bias[i];
        }
        return output;
    }

    double[] backward(double[] pdz,double learningRate){
        if(lastInput==null)
            throw new IllegalArgumentException("never called forward!");
        if(pdz.length!= weight.length)
            throw new IllegalArgumentException("pdz's length must match neurons' size");

        double[] pdx=new double[weight[0].length];
        //pdx 代表传入展开层的对于x的偏导数
        for(int c=0;c<weight.length;c++){
            for(int i=0;i< weight[0].length;i++){
                pdx[i]+=pdz[c]*weight[c][i];//计算对于x的偏导数
                weight[c][i]-=pdz[c]*lastInput[i]*learningRate;//更新这一层的权重
            }
            bias[c]-=pdz[c]*learningRate;//更新每一个神经元的偏移量
        }
        return pdx;
    }
}
