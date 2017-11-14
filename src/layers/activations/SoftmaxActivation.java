package layers.activations;

import layers.FlatLayer;
import math.Activations;
import math.Matrix;
import math.Vector;

/**
 * Activation Softmax, cette activation est un peu sp�cial donc elle ne rentre pas de le cadre de ActivationLayer car elle d�pend de toutes les entr�es.
 * A noter que la plupars du temps on pr�ferera utiliser SoftmaxCrossEntropy dans le cas ou la fonction de co�t utilis�e derri�re est EntropyLoss
 * et que la r�f�rence est un one-hot vector. (comme dans la plupart des probl�mes de classifications)
 */
public class SoftmaxActivation implements FlatLayer {
	Matrix cache;

	@Override
	public Matrix forward(Matrix in, boolean training) {
		in = Activations.softmax(in, Matrix.AXIS_HEIGHT);
		if (training)
			cache = new Matrix(in);
		return in;
	}

	@Override
	public Matrix backward(Matrix dout) {
		// La propagation arri�re est un peu insolite car elle part d'un code simple qui a �t� optimis�
		for (int i = 0; i < dout.width; i++) {
			Vector t = cache.get_column(i);
			double s = t.sum();
			for (int j = 0; j < dout.height; j++) {
				double v = t.v[j];
				dout.v[j][i] *= v * (-s + 2 * v - 1);
			}
		}
		return dout;
	}

	@Override
	public void apply_gradient() {
	}

	@Override
	public String toString() {
		return "SoftmaxActivation()";
	}
}
