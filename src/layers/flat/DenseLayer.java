package layers.flat;

import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Collection;

import layers.FlatLayer;
import layers.Parameters;
import layers.TrainableMatrices;
import layers.TrainableVectors;
import layers.activations.ActivationParser;
import math.Matrix;
import math.TrainableMatrix;
import math.TrainableVector;

/**
 * Classe permettant de simplifier l'écriture de réseau en regroupant 4 choses :
 * - Transformation affine
 * - Normalisation par batch
 * - Fonction d'activation
 * - Régularisation par Dropout
 */
public class DenseLayer implements FlatLayer, TrainableMatrices, TrainableVectors {
	public AffineLayer al;
	DropoutLayer dl;
	BatchnormLayer bl;
	public FlatLayer act;
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
	public Matrix backward(Matrix dout, boolean train) {
		// Pour la propagation arrière, on va dans l'autre sens
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
	
	public void write_to_file(String name) {
		this.al.write_to_file(name+"/al");
		if(batchnorm)
			this.bl.write_to_file(name+"/bl");
	}
	
	public void load_from_file(String name) throws FileNotFoundException {
		this.al.load_from_file(name+"/al");
		if(batchnorm)
			this.bl.load_from_file(name+"/bl");
	}

	@Override
	public String toString() {
		return al + "\n" + ((bl != null) ? bl + "\n" : "") + act + ((dl != null) ? "\n" + dl : "");
	}

	@Override
	public Collection<TrainableVector> get_trainable_vectors() {
		ArrayList<TrainableVector> ok = new ArrayList<>();
		ok.addAll(al.get_trainable_vectors());
		if(batchnorm)
			ok.addAll(bl.get_trainable_vectors());
		return ok;
	}

	@Override
	public Collection<TrainableMatrix> get_trainable_matrices() {
		return al.get_trainable_matrices();
	}
}
