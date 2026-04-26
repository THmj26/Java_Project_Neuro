//这里是将激活步骤分离出来 但是我觉得没有必要
//直接在卷积层就加入了激活步骤
public class activateLayer {
    private activateFuction af;
    activateLayer(activateFuction a){
        af=a;
    }

    public double[][] forward(double[][] input){
        double[][] output=new double[input.length][input[0].length];
        for(int i=0;i<input.length;i++){
            for(int j=0;j<input[0].length;j++){
                output[i][j]=af.activate(input[i][j]);
            }
        }
        return output;
    }
}
