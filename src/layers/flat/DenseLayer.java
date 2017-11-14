package layers.flat;

import layers.FlatLayer;
import layers.Parameters;
import layers.activations.ActivationLayer;
import layers.activations.ActivationParser;
import math.Matrix;

/**
 * Classe permettant de simplifier l'écriture de réseau en regroupant 4 choses :
 * - Transformation affine
 * - Normalisation par batch
 * - Fonction d'activation
 * - Régularisation par Dropout
 */
public class DenseLayer implements FlatLayer {
	public AffineLayer al;
	DropoutLayer dl;
	BatchnormLayer bl;
	ActivationLayer act;
	boolean dropout = false, batchnorm = false;

	/**
	 * @param fan_in dimension d'entrée
	 * @param fan_out dimension de sortie
	 * @param dropout taux de dropout
	 * @param activation nom de l'activation à utiliser
	 * @param batchnorm activer ou non la normalisation par batch
	 * @param p paramètres divers utilisé pour toutes les couches
	 */
	public DenseLayer(int fan_in, int fan_out, double dropout, String activation, boolean batchnorm, Parameters p) {
		al = new AffineLayer(fan_in, fan_out, true, p);
		if (batchnorm) {
			bl = new BatchnormLayer(fan_out, p);
			batchnorm = true;
		}
		act = ActivationParser.get_activation_by_name(activation);
		if (dropout > 0) {
			dl = new DropoutLayer(dropout);
			this.dropout = true;
		}

	}

	@Override
	public Matrix forward(Matrix in, boolean training) {
		// On effectue dans l'ordre précisée dans le commentaire de classe
		Matrix next = al.forward(in, training);
		if (batchnorm)
			next = bl.forward(next, training);
		if (act != null)
			next = act.forward(next, training);
		if (dropout)
			next = dl.forward(next, training);
		return next;
	}

	@Override
	public Matrix backward(Matrix dout) {
		// Pour la propagation arrière, on va dans l'autre sens
		Matrix next = dout;
		if (dropout)
			next = dl.backward(next);
		if (act != null)
			next = act.backward(next);
		if (batchnorm)
			next = bl.backward(next);
		next = al.backward(next);
		return next;
	}

	@Override
	public void apply_gradient() {
		al.apply_gradient();
		if (batchnorm)
			bl.apply_gradient();
		if (act != null)
			act.apply_gradient();
		if (dropout)
			dl.apply_gradient();
	}

	@Override
	public String toString() {
		return al + "\n" + ((bl != null) ? bl + "\n" : "") + act + ((dl != null) ? "\n" + dl : "");
	}
}
