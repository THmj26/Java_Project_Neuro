public class activateVectorLayer {
    private activateFuction af;
    activateVectorLayer(activateFuction a){
        af=a;
    }
    public double[] forward(double[] input){
        double[] output=new double[input.length];
        for(int i=0;i<input.length;i++){
            output[i]= af.activate(input[i]);
        }
        return output;
    }
}
