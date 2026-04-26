public class Sigmoid implements activateFuction{
    public double activate(double x){
        return 1.0/(1.0+Math.exp(-x));
    }
}
