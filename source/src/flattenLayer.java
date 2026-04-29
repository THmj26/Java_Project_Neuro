public class flattenLayer {
        int lastI;
        int lastR;
        int lastC;
        public double[] forward(double[][][] input) {
            lastI = input.length;        // 图片数量
            lastR = input[0].length;     // 每张图的行数
            lastC = input[0][0].length;  // 每张图的列数

            double[] output = new double[lastI * lastR * lastC];

            int index = 0;
            //将所有图片的所有像素展开到一个一维数组
            for (double[][] doubles : input) {
                for (int i = 0; i < lastR; i++) {
                    for (int j = 0; j < lastC; j++) {
                        output[index] = doubles[i][j];
                        index++;
                    }
                }
            }
            return output;
        }
    //在这一层中的backward中不做任何计算 就是将之前flat后的数据 再重新变回矩阵去
        public double[][][] backward(double[] pdx){
            if(pdx.length!=lastC*lastR*lastI)
                throw new IllegalArgumentException("pdx's length doesn't match flatten input/output size");
            double[][][] gradient=new double[lastI][lastR][lastC];
            int index=0;
            for(int c=0;c<lastI;c++){
                for(int i=0;i<lastR;i++){
                    for(int j=0;j<lastC;j++){
                        gradient[c][i][j]=pdx[index];
                        index++;
                    }
                }
            }
            return gradient;
        }
}
