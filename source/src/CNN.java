public class CNN {
    private convLayer cL;
    private activateLayer aL;
    private poolingLayer pL;
    private flattenLayer fL;
    private denseLayer dL;
    private activateVectorLayer avL;

    CNN(convLayer cL,activateLayer aL,poolingLayer pL,flattenLayer fL,denseLayer dL, activateVectorLayer avl){
        this.cL=cL;
        this.aL=aL;
        this.pL=pL;
        this.fL=fL;
        this.dL=dL;
        this.avL=avl;
    }
    public double[] forward(double[][][] input){
        double[][][] convOutput = cL.forward(input);
        double[][][] activatedConv = aL.forward(convOutput);
        double[][][] pooled = pL.forward(activatedConv);
        double[] flat = fL.forward(pooled);
        double[] denseOutput = dL.forward(flat);
        return avL.forward(denseOutput);
    }
}
