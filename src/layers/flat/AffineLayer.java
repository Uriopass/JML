package layers.flat;

import layers.FlatLayer;
import layers.Parameters;
import math.Initialisations;
import math.Matrix;
import math.Optimizers;
import math.RMSMatrix;
import math.RMSVector;

/**
 * Cette classe correspond à une transformation affine avec biais d'un espace de fan_in dimensions vers fan_out dimensions.
 * On peut donc représenter cela par une matrice W et un vecteur b tel que 
 * - y = Wx + b
 */
public class AffineLayer implements FlatLayer {
	// Matrice de poids
	public RMSMatrix weight;
	// Vecteur de biais
	public RMSVector bias;

	private Matrix cache;

	// Régularisation L2
	public double regularization;
	// Taux d'apprentissage
	public double learning_rate;
	// Decay d'apprentissage à la fin de chaque époque
	public double learning_rate_decay;
	// gamma pour l'optimisation RMSProp
	public double gamma;
	// epsilon pour l'optimisation RMSProp
	public double epsilon;
	// permet d'éviter de calculer la dérivée par rapport aux données pour la première couche
	public boolean calculate_dout;

	// Dimensions d'entrée et de sorties
	public int fan_in, fan_out;

	/**
	 * @param fan_in Dimension d'entrée
	 * @param fan_out Dimension de sorties
	 * @param init Inititialisation ou non des poids de la couche (ici l'initialisation xavier est utilisée), variance = sqrt(2 / (fan_in + fan_out))
	 * @param p paramètres optionnels.
	 * Par défaut - Nom - Description
	 * 0.001 - lr - taux d'apprentissage
	 * 0 - reg - taux de regularisation L2
	 * 1 -lrdecay - decay du taux d'apprentissage
	 * 0.9 - gamma - utilisé pour RMSProp
	 * 1e-6 - epsilon - utilisé pour RMSProp
	 * true - dout - calculer dout ou non
	 */
	public AffineLayer(int fan_in, int fan_out, boolean init, Parameters p) {
		this.fan_in = fan_in;
		this.fan_out = fan_out;
		weight = new RMSMatrix(fan_in, fan_out);
		bias = new RMSVector(fan_out);

		if (init) {
			Initialisations.xavier(weight, fan_in, fan_out);
			bias.fill(0);
		}

		if (p == null) {
			p = new Parameters();
		}

		this.regularization = p.get_as_double("reg", 0);
		this.learning_rate = p.get_as_double("lr", 0.001);
		this.learning_rate_decay = p.get_as_double("lrdecay", 1);
		this.gamma = p.get_as_double("gamma", 0.9);
		this.epsilon = p.get_as_double("epsilon", 1e-6);
		this.calculate_dout = p.get_or_default("dout", "true").equalsIgnoreCase("true");
	}

	public void end_of_epoch() {
		learning_rate *= learning_rate_decay;
	}

	@Override
	public Matrix forward(Matrix in, boolean training) {
		if (training)
			cache = new Matrix(in);
		Matrix next = weight.parralel_mult(in);
		next.add(bias, Matrix.AXIS_WIDTH);
		return next;
	}

	@Override
	public Matrix backward(Matrix dout) {
		// A noter que pour aller plus vite, on parralélise la multiplication de matrice avec parralel_mult

		// Propagation arrière :

		// Ici dout correspond aux dérivées partielles sur y
		// w' = dout*(x.T) / mini_batch + reg*w
		// b' = moyennes des valeurs de dout sur l'axe du mini batch
		// x' = (w.T)*dout

		weight.grad.add(dout.parralel_mult(cache.T()).scale(1.0 / cache.width));
		if(regularization != 0)
			weight.grad.add(Matrix.scale(weight, regularization));
		bias.grad.add(dout.sum(Matrix.AXIS_WIDTH).scale(1.0 / cache.width));

		if (!calculate_dout)
			return null;

		return weight.T().parralel_mult(dout);
	}

	@Override
	public void apply_gradient() {
		/*
		weight.add(weight.grad.scale(-learning_rate));
		weight.grad.fill(0);
		bias.add(bias.grad.scale(-learning_rate));
		bias.grad.fill(0);
		*/
		// On utilise RMSProp pour l'optimisation, car la convergence est de meileure qualitée qu'une simple descente de gradient
		Optimizers.RMSProp(weight, gamma, learning_rate, epsilon);
		Optimizers.RMSProp(bias, gamma, learning_rate, epsilon);
	}

	@Override
	public String toString() {
		return "AffineLayer(" + fan_in + ", " + fan_out + ", lr=" + learning_rate + ", reg=" + regularization
				+ ", lrdecay=" + learning_rate_decay + ((calculate_dout) ? "" : ", dout=false") + ")";
	}
}
