package layers.flat;

import java.io.FileNotFoundException;
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
 * Cette classe correspond à une transformation affine avec biais d'un espace de fan_in dimensions vers fan_out dimensions.
 * On peut donc représenter cela par une matrice W et un vecteur b tel que 
 * - y = Wx + b
 */
public class AffineLayer implements FlatLayer, TrainableMatrices, TrainableVectors {
	private Matrix cache;
	
	// permet d'éviter de calculer la dérivée par rapport aux données pour la première couche
	public boolean calculate_dout;

	public HashMap<String, TrainableMatrix> matrices = new HashMap<>();
	public HashMap<String, TrainableVector> vectors = new HashMap<>();

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
	 * true - dout - calculer dout ou non
	 */
	public AffineLayer(int fan_in, int fan_out, boolean init, Parameters p) {
		this.fan_in = fan_in;
		this.fan_out = fan_out;
		
		matrices.put("w", new TrainableMatrix(fan_in, fan_out));

		vectors.put("b", new TrainableVector(fan_out));

		if (init) {
			//TODO
			//Initialisations.xavier(matrices.get("w"), fan_in, fan_out);
			Initialisations.gaussian(matrices.get("w"),0.01);
		}

		if (p == null) {
			p = new Parameters();
		}

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

	/**
	 * Use this with precaution
	 */
	public void setCache(Matrix cache) {
		this.cache = cache;
	}
	
	@Override
	public Matrix backward(Matrix dout, boolean train) {
		// A noter que pour aller plus vite, on parralélise la multiplication de matrice avec parralel_mult

		// Propagation arrière :

		// Ici dout correspond aux dérivées partielles sur y
		// w' = dout*(x.T) / mini_batch + reg*w
		// b' = moyennes des valeurs de dout sur l'axe du mini batch
		// x' = (w.T)*dout
		if(train) {	
			matrices.get("w").grad.add(dout.parralel_mult(cache.T()).scale(1.0 / cache.width));
			vectors.get("b").grad.add(dout.sum(Matrix.AXIS_WIDTH).scale(1.0 / cache.width));
		}
		
		if (!calculate_dout)
			return null;

		return Matrix.parralel_mult(matrices.get("w").T(), dout);
	}
	
	public TrainableMatrix get_weight() {
		return matrices.get("w");
	}

	public TrainableVector get_bias() {
		return vectors.get("b");
	}
	
	public void write_to_file(String name) {
		matrices.get("w").write_to_file(name+"/w");
		vectors.get("b").to_row_matrix().write_to_file(name+"/b");
	}
	
	public void load_from_file(String name) throws FileNotFoundException {
		matrices.get("w").load_from_file(name+"/w");
		vectors.get("b").replace_by(Matrix.generate_from_file(name+"/b").get_row(0));
	}

	@Override
	public String toString() {
		return "AffineLayer(" + fan_in + ", " + fan_out
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
