public class MseLoss {
    public double loss(double[] target,double[] predict){
        double sum=0;
        for(int i=0;i<predict.length;i++){
            sum+=Math.pow(target[i]-predict[i],2);
        }
        return sum/2;//和之前NN一样 都是为了求导数方便
    }
    public double[] gradient(double[] target,double[] predict){
        double[] g=new double[predict.length];
        for(int i=0;i< predict.length;i++){
            g[i]=predict[i]-target[i];
        }
        return g;
    }
}
