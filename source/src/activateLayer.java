//import javax.swing.*;

//这里是对所有的卷积结果图片进行激活
//在这里引入激活层是因为 每一个神经元一个激活函数太麻烦
//而且一般而言同一层中的神经元的激活函数是一致的
public class activateLayer {
    private activateFuction af;
    private double[][][] lastInput;

    activateLayer(activateFuction a){
        af=a;
    }

    public double[][][] forward(double[][][] input){
        lastInput=input.clone();
        int pictureI=input.length;//图片数量
        int pictureR=input[0].length;//单张图片的行数
        int pictureC=input[0][0].length;//列数

        double[][][] output=new double[pictureI][pictureR][pictureC];
        //对每一张图片中的每一个顶点上的数据进行激活
        //外层是对图片遍历
        for(int c=0;c<pictureI;c++){
            for(int i=0;i<pictureR;i++){
                for(int j=0;j<pictureC;j++){
                    output[c][i][j]=af.activate(input[c][i][j]);
                }
            }
        }
        return output;
    }
    public double[][][] backward(double[][][] pdx){
        int c= pdx.length;
        int R=pdx[0].length;
        int C=pdx[0][0].length;
        double[][][] gradient=new double[c][R][C];
        for(int p=0;p<c;p++){
            for(int i=0;i<R;i++){
                for(int j=0;j<C;j++){
                    gradient[p][i][j]=af.derivative(lastInput[p][i][j])*pdx[p][i][j];
                }
            }
        }
        return gradient;
    }
}
