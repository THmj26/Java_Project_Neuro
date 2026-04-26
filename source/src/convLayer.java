//层结构类似于之前NN中的DenseLayer
public class convLayer {
    convNeuron[] filters;
    convLayer(convNeuron[] c){
        filters =c;
    }

    double[][][] forward(double[][][] input){
        double[][][] output=new double[filters.length][][];
        for(int i=0;i<filters.length;i++){
            output[i]=filters[i].forward(input);
        }
        return output;
    }
}
