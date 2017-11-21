package layers.losses;

import math.Matrix;

/**
 * Fonction de coût d'entropie croisée
 * avec x la prédiction et y la référence (probabilitées)
 * L(x, y) = (1-y)*ln(1-x) + y*ln(x)
 * L'(x, y) = (y-x)/(x*(1-x))
 */
public class EntropyLoss extends Loss {
	@Override
	public Matrix forward(Matrix in, boolean training) {
		return in;
	}

	@Override
	public Matrix backward(Matrix dout, boolean train) {
		loss = 0;
		for (int i = 0; i < dout.height; i++) {
			for (int j = 0; j < dout.width; j++) {
				double y = refs.v[i][j];
				double y_prime = dout.v[i][j];
				loss -= (1 - y) * Math.log(Math.max(1e-10, 1 - y_prime)) + y * Math.log(Math.max(1e-10, y_prime));
				dout.v[i][j] = (y - y_prime) / (y_prime - y_prime * y_prime);
			}
		}
		loss /= dout.width;

		return dout;
	}

	@Override
	public String toString() {
		return "EntropyLoss()";
	}
}
