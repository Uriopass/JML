package layers;

public class ReLUActivation extends ActivationLayer {
	@Override
	public double activation_forward(double in) {
		return Math.max(0, in);
	}

	@Override
	public double activation_backward(double in) {
		int sig = (int) Math.signum(in);
		return (sig+1)/2;
	}
	
	@Override
	public String toString() {
		return "TanhActivation()";
	}
}
