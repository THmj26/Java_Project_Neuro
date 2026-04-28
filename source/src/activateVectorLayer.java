public class activateVectorLayer {
    private activateFuction af;
    private double[] lastInput;

    activateVectorLayer(activateFuction a){
        af=a;
    }

    public double[] forward(double[] input){
        lastInput =input.clone();
        double[] output=new double[input.length];
        for(int i=0;i<input.length;i++){
            output[i]= af.activate(input[i]);
        }
        return output;
    }

    public double[] backward(double[] pdy){
        double[] g=new double[pdy.length];
        for(int i=0;i< pdy.length;i++){
            g[i]=af.derivative(lastInput[i])*pdy[i];
        }
        return g;
    }
}
