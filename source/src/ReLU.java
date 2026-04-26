public class ReLU implements activateFuction {
    public double activate(double x){
        return Math.max(0,x);
    }
}
