package layers.losses;

import layers.FlatLayer;
import math.Matrix;

/**
 * Classe générique d'une fonction de coût, qui possède donc la référence et le coût calculé dans backward()
 * Les références sont passé grâce à feed_ref, qui prend en argument une matrice de vérité.
 */
public abstract class Loss implements FlatLayer {
	// Matrice de vérité
	Matrix refs;
	// Coût calculé dans backward()
	public double loss;

	public void feed_ref(Matrix ref) {
		this.refs = ref;
	}

	public static Matrix from_int_refs(int[] refs, int nb_class) {
		Matrix m = new Matrix(refs.length, nb_class);
		for (int i = 0; i < refs.length; i++) {
			m.v[refs[i]][i] = 1;
		}
		return m;
	}

	@Override
	public void apply_gradient() {
	}
}
