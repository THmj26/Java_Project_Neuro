public class flattenLayer {

    public double[] forward(double[][][] input) {
        int pictureI = input.length;        // 图片数量
        int pictureR = input[0].length;     // 每张图的行数
        int pictureC = input[0][0].length;  // 每张图的列数

        double[] output = new double[pictureI * pictureR * pictureC];

        int index = 0;
        //将所有图片的所有像素展开到一个一维数组
        for (double[][] doubles : input) {
            for (int i = 0; i < pictureR; i++) {
                for (int j = 0; j < pictureC; j++) {
                    output[index] = doubles[i][j];
                    index++;
                }
            }
        }

        return output;
    }
}
