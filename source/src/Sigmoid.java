public class Sigmoid implements activateFuction{
    public double activate(double x){
        return 1.0/(1.0+Math.exp(-x));
    }
    public double derivative(double z){
        return (1.0/(1.0+Math.exp(-z)))*(1-(1.0/(1.0+Math.exp(-z))));
    }
}
