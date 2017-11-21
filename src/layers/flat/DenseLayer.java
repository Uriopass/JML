package layers.flat;

import layers.FlatLayer;
import layers.Parameters;
import layers.activations.ActivationLayer;
import layers.activations.ActivationParser;
import math.Matrix;

/**
 * Classe permettant de simplifier l'�criture de r�seau en regroupant 4 choses :
 * - Transformation affine
 * - Normalisation par batch
 * - Fonction d'activation
 * - R�gularisation par Dropout
 */
public class DenseLayer implements FlatLayer {
	public AffineLayer al;
	DropoutLayer dl;
	BatchnormLayer bl;
	ActivationLayer act;
	boolean dropout = false, batchnorm = false;

	/**
	 * @param fan_in dimension d'entr�e
	 * @param fan_out dimension de sortie
	 * @param dropout taux de dropout
	 * @param activation nom de l'activation � utiliser
	 * @param batchnorm activer ou non la normalisation par batch
	 * @param p param�tres divers utilis� pour toutes les couches
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
		// On effectue dans l'ordre pr�cis�e dans le commentaire de classe
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
	public Matrix backward(Matrix dout, boolean train) {
		// Pour la propagation arri�re, on va dans l'autre sens
		Matrix next = dout;
		if (dropout)
			next = dl.backward(next, train);
		if (act != null)
			next = act.backward(next, train);
		if (batchnorm)
			next = bl.backward(next, train);
		next = al.backward(next, train);
		return next;
	}

	@Override
	public String toString() {
		return al + "\n" + ((bl != null) ? bl + "\n" : "") + act + ((dl != null) ? "\n" + dl : "");
	}
}
