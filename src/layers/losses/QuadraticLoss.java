package layers.losses;

import math.Matrix;

/**
 * Fonction de coût d'erreur quadratique
 * L(x, y) = 0.5*(x-y)^2
 */
public class QuadraticLoss extends Loss {

	Matrix ref;
	public double loss;

	@Override
	public Matrix forward(Matrix in, boolean training) {
		return in;
	}

	@Override
	public Matrix backward(Matrix dout, boolean train) {
		dout.add(ref.scale(-1));
		loss = 0.5 * Matrix.hadamart(dout, dout).sum() / dout.width;
		return dout;
	}

	@Override
	public String toString() {
		return "QuadraticLoss()";
	}
}
