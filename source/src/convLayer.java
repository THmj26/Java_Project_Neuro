//层结构类似于之前NN中的DenseLayer
public class convLayer {
    convNeuron[] filters;
    convLayer(convNeuron[] c){
        filters =c;
    }

    public double[][][] forward(double[][][] input){
        double[][][] output=new double[filters.length][][];
        for(int i=0;i<filters.length;i++){
            output[i]=filters[i].forward(input);
        }
        return output;
    }
    public double[][][] backward(double[][][] pdz,double learningRate){
           double[][][] gradient=null;
           for(int i=0;i<filters.length;i++){
               double[][][] oneNeuronGradient=filters[i].backward(pdz[i],learningRate);
               if(gradient==null)
                   gradient=new double[oneNeuronGradient.length][oneNeuronGradient[0].length][oneNeuronGradient[0][0].length];
               for(int p=0;p<oneNeuronGradient.length;p++){
                   for(int k=0;k<oneNeuronGradient[0].length;k++){
                       for(int j=0;j<oneNeuronGradient[0][0].length;j++){
                           gradient[p][k][j]+=oneNeuronGradient[p][i][j];
                       }
                   }
               }
           }
           return gradient;
    }
}
