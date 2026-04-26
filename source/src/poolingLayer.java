public class poolingLayer {
    private int poolSize;
    private int stride;

    poolingLayer(int size,int stride){
        poolSize=size;
        this.stride=stride;
    }
    public double[][][] forward(double[][][] input){
        int pictureI=input.length;//图片数量
        int pictureR=input[0].length;//单张图片的行数
        int pictureC=input[0][0].length;//列数
        //这里对于池化层 输出图片的大小由 框的大小和步长共同决定
        //前半部分是算能往后移动几步 加一表示的是加上一开始的那个框
        int outR=(pictureR-poolSize)/stride+1;
        int outC=(pictureC-poolSize)/stride+1;
        double[][][] output=new double[pictureI][outR][outC];
        //池化的原理就是对一个框框内的元素取最大值 而后返回图片
        //本质上就是一个压缩过程
        for(int c=0;c<pictureI;c++){
            for(int i=0;i<outR;i++){
                for(int j=0;j<outC;j++){

                    double max=Double.NEGATIVE_INFINITY;
                    for(int ki=0;ki<poolSize;ki++){
                        for(int kj=0;kj<poolSize;kj++){
                            int R=i*stride+ki;
                            int C=j*stride+kj;

                            max=Math.max(max,input[c][R][C]);
                        }
                    }
                    output[c][i][j]=max;

                }
            }
        }
        return output;
    }
}
