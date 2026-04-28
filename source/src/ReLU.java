public class ReLU implements activateFuction {
    public double activate(double x){
        return Math.max(0,x);
    }
    public double derivative(double z){
        if (z>0)
            return 1;
        else
            return 0;
    }
}
