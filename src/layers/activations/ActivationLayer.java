package layers.activations;

import layers.FlatLayer;
import math.Matrix;
import math.Vector;

/**
 * Cette classe permet une généralisation des fonctions d'activation, en simplifiant toutes les boucles en 2 fonctions très simples :
 *  - activation_forward(double in);
 *  - activation_backward(); qui utilise les fonctions get_before() et get_after()
 *  
 *  Si y = activation(x), alors get_before() = x et get_after() = y
 */
public abstract class ActivationLayer implements FlatLayer {
	// Indique quelles données sont utile pour la propagation arrière
	protected boolean needs_cache_before = false;
	protected boolean needs_cache_after = false;

	public abstract double activation_forward(double in);

	public abstract double activation_backward();

	/// Caches pour le calcul de la propagation arrière
	// Le cache "before" correspond aux données avant l'activation
	private Matrix m_cache_before;
	// Le cache "after"  correspond aux données après l'activation
	private Matrix m_cache_after;

	private Vector v_cache_before;
	private Vector v_cache_after;

	// Indices utilisé en interne pour get_before() et get_after()
	private int i, j;

	protected double get_before() {
		if (m_cache_before != null)
			return m_cache_before.v[i][j];
		if (v_cache_before != null)
			return v_cache_before.v[i];
		throw new RuntimeException("Asking for before, but no before found");
	}

	protected double get_after() {
		if (m_cache_after != null)
			return m_cache_after.v[i][j];
		if (v_cache_after != null)
			return v_cache_after.v[i];
		throw new RuntimeException("Asking for after, but no after found");
	}

	@Override
	public Matrix forward(Matrix in, boolean training) {

		if (training && needs_cache_before)
			m_cache_before = new Matrix(in);

		for (int i = 0; i < in.height; i++) {
			for (int j = 0; j < in.width; j++) {
				in.v[i][j] = activation_forward(in.v[i][j]);
			}
		}
		if (training && needs_cache_after)
			m_cache_after = new Matrix(in);
		return in;
	}

	public Vector forward(Vector in, boolean training) {
		if (training && needs_cache_before)
			v_cache_before = new Vector(in);
		for (int i = 0; i < in.length; i++) {
			in.v[i] = activation_forward(in.v[i]);
		}
		if (training && needs_cache_after)
			v_cache_after = new Vector(in);
		return in;
	}

	@Override
	public Matrix backward(Matrix dout, boolean train) {
		for (int i = 0; i < dout.height; i++) {
			this.i = i;
			for (int j = 0; j < dout.width; j++) {
				this.j = j;
				dout.v[i][j] *= activation_backward();
			}
		}
		return dout;
	}

	public Vector backward(Vector dout) {
		for (int i = 0; i < dout.length; i++) {
			this.i = i;
			dout.v[i] *= activation_backward();
		}
		return dout;
	}
}
