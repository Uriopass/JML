package layers.flat;

import layers.FlatLayer;
import layers.Parameters;
import math.Initialisations;
import math.Matrix;
import math.Optimizers;
import math.RMSVector;

/**
 * Cette classe est un test de couche personalisé, l'idée est de ne connecter chaque neurone d'entrée qu'a 1 seul neurone de sorties, de manière uniforme.
 * Par exemple si on prends fan_in = 100, fan_out = 10 alors les 10 premiers neurones d'entrée ne seront connecté qu'au premier neurone de sortie.
 * Dans le cas de MNIST par exemple cela forcerait les 10 premiers neurones à se spécialiser dans la reconnaissance du chiffre 0.
 */
public class SplitAffineLayer implements FlatLayer {
	public RMSVector weight;
	public RMSVector bias;

	private Matrix cache;

	public double regularization;
	public double learning_rate;
	public double learning_rate_decay;
	public double gamma;
	public double epsilon;

	public boolean calculate_dout;

	public int fan_in, fan_out, per_out;

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
	public SplitAffineLayer(int fan_in, int fan_out, boolean init, Parameters p) {
		if (fan_in % fan_out != 0) {
			throw new RuntimeException("fan_in (" + fan_in + ") must be a multiple of fan_out (" + fan_out + ")");
		}
		this.fan_in = fan_in;
		this.fan_out = fan_out;
		this.per_out = fan_in / fan_out;

		weight = new RMSVector(fan_in);
		bias = new RMSVector(fan_out);

		if (init) {
			Initialisations.he_uniform(weight, per_out, p.get_as_double("init_multiplier", 1));

			for (int i = 0; i < bias.length; i++) {
				bias.v[i] = 0;
			}
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

		Matrix next = new Matrix(in.width, fan_out);
		in.scale(weight, Matrix.AXIS_WIDTH);
		for (int k = 0; k < fan_out; k++) {
			for (int i = 0; i < in.width; i++) {
				double sum = 0;
				for (int j = 0; j < per_out; j++) {
					sum += in.v[j + k * per_out][i];
				}
				next.v[k][i] = sum;
			}
		}

		next.add(bias, Matrix.AXIS_WIDTH);
		return next;
	}

	@Override
	public Matrix backward(Matrix dout) {
		bias.grad.add(dout.sum(Matrix.AXIS_WIDTH).scale(1.0 / cache.width));

		Matrix expanded_dout = new Matrix(cache.width, cache.height);

		for (int i = 0; i < expanded_dout.height; i++) {
			for (int j = 0; j < expanded_dout.width; j++) {
				expanded_dout.v[i][j] = dout.v[i / per_out][j];
			}
		}
		weight.grad.add(Matrix.hadamart(expanded_dout, cache).sum(Matrix.AXIS_WIDTH).scale(1.0 / cache.width));
		if (!calculate_dout)
			return null;
		return expanded_dout.scale(weight, Matrix.AXIS_WIDTH);
	}

	@Override
	public void apply_gradient() {
		Optimizers.RMSProp(weight, gamma, learning_rate, epsilon);
		Optimizers.RMSProp(bias, gamma, learning_rate, epsilon);
	}

	@Override
	public String toString() {
		return "SplitAffineLayer(" + fan_in + ", " + fan_out + ", lr=" + learning_rate + ", reg=" + regularization
				+ ", lrdecay=" + learning_rate_decay + ((calculate_dout) ? "" : ", dout=false") + ")";
	}
}
