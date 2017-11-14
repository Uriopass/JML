package layers.activations;

/**
 * Cette classe permet de transformer une string en activation, utile quand l'activation est un paramètre comme dans DenseLayer()
 */
public class ActivationParser {
	public static ActivationLayer get_activation_by_name(String name) {
		switch (name.toLowerCase()) {
		case "tanh":
			return new TanhActivation();
		case "sigmoid":
		case "sig":
			return new SigmoidActivation();
		case "relu":
			return new ReLUActivation();
		case "swish":
			return new SwishActivation();
		case "softplus":
			return new SoftplusActivation();
		default:
			return null;
		}
	}
}
