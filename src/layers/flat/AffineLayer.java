package layers.flat;

import java.util.Collection;
import java.util.HashMap;

import layers.FlatLayer;
import layers.Parameters;
import layers.TrainableMatrices;
import layers.TrainableVectors;
import math.Initialisations;
import math.Matrix;
import math.TrainableMatrix;
import math.TrainableVector;

/**
 * Cette classe correspond � une transformation affine avec biais d'un espace de fan_in dimensions vers fan_out dimensions.
 * On peut donc repr�senter cela par une matrice W et un vecteur b tel que 
 * - y = Wx + b
 */
public class AffineLayer implements FlatLayer, TrainableMatrices, TrainableVectors {
	private Matrix cache;

	// R�gularisation L2
	public double regularization;
	// permet d'�viter de calculer la d�riv�e par rapport aux donn�es pour la premi�re couche
	public boolean calculate_dout;

	public HashMap<String, TrainableMatrix> matrices = new HashMap<>();
	public HashMap<String, TrainableVector> vectors = new HashMap<>();

	// Dimensions d'entr�e et de sorties
	public int fan_in, fan_out;

	/**
	 * @param fan_in Dimension d'entr�e
	 * @param fan_out Dimension de sorties
	 * @param init Inititialisation ou non des poids de la couche (ici l'initialisation xavier est utilis�e), variance = sqrt(2 / (fan_in + fan_out))
	 * @param p param�tres optionnels.
	 * Par d�faut - Nom - Description
	 * 0.001 - lr - taux d'apprentissage
	 * 0 - reg - taux de regularisation L2
	 * 1 -lrdecay - decay du taux d'apprentissage
	 * true - dout - calculer dout ou non
	 */
	public AffineLayer(int fan_in, int fan_out, boolean init, Parameters p) {
		this.fan_in = fan_in;
		this.fan_out = fan_out;
		
		matrices.put("w", new TrainableMatrix(fan_in, fan_out));

		vectors.put("b", new TrainableVector(fan_out));

		if (init) {
			Initialisations.xavier(matrices.get("w"), fan_in, fan_out);
		}

		if (p == null) {
			p = new Parameters();
		}

		this.regularization = p.get_as_double("reg", 0);
		this.calculate_dout = p.get_or_default("dout", "true").equalsIgnoreCase("true");
	}

	@Override
	public Matrix forward(Matrix in, boolean training) {
		if (training)
			cache = new Matrix(in);
		Matrix next = matrices.get("w").parralel_mult(in);
		next.add(vectors.get("b"), Matrix.AXIS_WIDTH);
		return next;
	}

	
	
	@Override
	public Matrix backward(Matrix dout, boolean train) {
		// A noter que pour aller plus vite, on parral�lise la multiplication de matrice avec parralel_mult

		// Propagation arri�re :

		// Ici dout correspond aux d�riv�es partielles sur y
		// w' = dout*(x.T) / mini_batch + reg*w
		// b' = moyennes des valeurs de dout sur l'axe du mini batch
		// x' = (w.T)*dout
		if(train) {	
			matrices.get("w").grad.add(dout.parralel_mult(cache.T()).scale(1.0 / cache.width));
			if(regularization != 0)
				matrices.get("w").grad.add(Matrix.scale(matrices.get("w"), regularization));
			vectors.get("b").grad.add(dout.sum(Matrix.AXIS_WIDTH).scale(1.0 / cache.width));
		}
		
		if (!calculate_dout)
			return null;

		return matrices.get("w").T().parralel_mult(dout);
	}

	@Override
	public String toString() {
		return "AffineLayer(" + fan_in + ", " + fan_out + ", reg=" + regularization
			 + ((calculate_dout) ? "" : ", dout=false") + ")";
	}

	@Override
	public Collection<TrainableMatrix> get_trainable_matrices() {
		return matrices.values();
	}

	@Override
	public Collection<TrainableVector> get_trainable_vectors() {
		return vectors.values();
	}
}
