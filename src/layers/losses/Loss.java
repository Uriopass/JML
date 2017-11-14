package layers.losses;

import layers.FlatLayer;
import math.Matrix;

/**
 * Classe g�n�rique d'une fonction de co�t, qui poss�de donc la r�f�rence et le co�t calcul� dans backward()
 * Les r�f�rences sont pass� gr�ce � feed_ref, qui prend en argument une matrice de v�rit�.
 */
public abstract class Loss implements FlatLayer {
	// Matrice de v�rit�
	Matrix refs;
	// Co�t calcul� dans backward()
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
