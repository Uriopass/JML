package math;

/**
 * La classe Vector est une surcouche d'un tableau de double, et permet de grandement simplifier la gestion de vecteur, rendant le code plus lisible.
 */
public class Vector {

	// Taille du vectuer
	public int length;

	// Données du vecteur
	public double[] v;

	/**
	 * Crée un vecteur à partir d'une ensemble de données
	 */
	public Vector(double[] data) {
		v = data;
		length = v.length;
	}

	/**
	 * Crée un vecteur remplie de zéros de longueur length
	 * @param length taille du vecteur
	 */
	public Vector(int length) {
		v = new double[length];
		this.length = length;
	}

	/**
	 * Crée un vecteur à partir d'un ensemble de données entières
	 */
	public Vector(int[] data) {
		length = data.length;
		v = new double[length];
		for (int i = 0; i < length; i++) {
			v[i] = data[i];
		}
	}

	/**
	 * Crée un vecteur à partir d'un autre vecteur
	 * @param v vecteur à copier
	 */
	public Vector(Vector v) {
		this.v = new double[v.length];
		this.length = v.length;
		for (int i = 0; i < length; i++)
			this.v[i] = v.v[i];
	}
	
	public static Vector one_hot(int length, int position) {
		Vector v = new Vector(length);
		v.v[position] = 1;
		return v;
	}

	/**
	 * Ajoute un scalaire à ce vecteur
	 * @param val scalaire à ajouter
	 * @return this
	 */
	public Vector add(double val) {
		for (int i = 0; i < length; i++) {
			v[i] += val;
		}
		return this;
	}

	/**
	 * Ajoute un vecteur à ce vecteur
	 * @param b vecteur à ajouter
	 * @return this
	 */
	public Vector add(Vector b) {
		if (length != b.length) {
			throw new RuntimeException("Incompatible shape : (" + length + ") and (" + b.length + ")");
		}
		for (int i = 0; i < length; i++) {
			v[i] += b.v[i];
		}
		return this;
	}

	/**
	 * Ajoute deux vecteurs ensemble
	 * @param a premier vecteur à ajouter
	 * @param b second vecteur à ajouter
	 * @return nouveau vecteur contenant la somme des deux
	 */
	public static Vector add(Vector a, Vector b) {
		if (a.length != b.length) {
			throw new RuntimeException("Incompatible shape : (" + a.length + ") and (" + b.length + ")");
		}
		Vector res = new Vector(a.length);
		for (int i = 0; i < a.length; i++) {
			res.v[i] = a.v[i] + b.v[i];
		}
		return res;
	}

	/**
	 * Concatène un vecteur à ce vecteur
	 * @param b vecteur à mettre en bout
	 * @return this conc b
	 */
	public Vector append(Vector b) {
		int aLen = length;
		int bLen = b.length;
		double[] c = new double[aLen + bLen];
		System.arraycopy(v, 0, c, 0, aLen);
		System.arraycopy(b.v, 0, c, aLen, bLen);
		v = c;
		this.length = aLen + bLen;
		return this;
	}

	/**
	 * Renvoie l'indice de la plus grande valeur de ce vecteur
	 */
	public int argmax() {
		int max = 0;
		double maxV = v[0];
		for (int i = 1; i < v.length; i++) {
			if (v[i] > maxV) {
				maxV = v[i];
				max = i;
			}
		}
		return max;
	}

	public int count_zeros() {
		int zeros = 0;
		for(int i = 0 ; i < length ; i++) {
			if(v[i] == 0)
				zeros += 1;
		}
		return zeros;
	}
	
	/**
	 * Effectue le produit scalaire de ce vecteur par un autre vecteur
	 * @param b
	 * @return
	 */
	public double dot(Vector b) {
		double sum = 0;
		for (int i = 0; i < length; i++) {
			sum += v[i] * b.v[i];
		}
		return sum;
	}

	/**
	 * Remplie ce vecteur avec une valeur
	 */
	public void fill(double value) {
		for (int i = 0; i < v.length; i++) {
			v[i] = value;
		}
	}

	/**
	 * Inverse ce vecteur, élément-à-élement
	 * @return v_i = 1 / v_i
	 */
	public Vector inverse() {
		for (int i = 0; i < length; i++) {
			v[i] = 1.0 / v[i];
		}
		return this;
	}

	/**
	 * Renvoie le maximum de ce vecteur
	 */
	public double max() {
		double max = v[0];
		for (int i = 1; i < v.length; i++)
			max = Math.max(max, v[i]);
		return max;
	}

	/**
	 * Renvoie le minimum de ce vecteur
	 */
	public double min() {
		double min = v[0];
		for (int i = 1; i < v.length; i++)
			min = Math.min(min, v[i]);
		return min;
	}

	/**
	 * Effectue le produit externe de deux vecteurs
	 * @param a vecteur en colonne
	 * @param b vecteur en ligne
	 * @return m_ij = a_i * b_j
	 */
	public static Matrix outer(Vector a, Vector b) {
		Matrix m = new Matrix(b.length, a.length);
		for (int i = 0; i < m.height; i++) {
			for (int j = 0; j < m.width; j++) {
				m.v[i][j] = a.v[i] * b.v[j];
			}
		}
		return m;
	}

	/**
	 * Mets ce vecteur à la puissance pow, élement-à-élement
	 * @param pow puissance à utiliser
	 * @return this
	 */
	public Vector power(double pow) {
		if (pow == 0.5) {
			for (int i = 0; i < length; i++) {
				v[i] = Math.sqrt(v[i]);
			}
		} else {
			for (int i = 0; i < length; i++) {
				v[i] = Math.pow(v[i], pow);
			}
		}
		return this;
	}

	/**
	 * Met un vecteur à la puissance pow, élément à élement
	 * @param a vecteur à mettre en puissance
	 * @param pow puissance à utiliser
	 * @return a_i = a_i^pow
	 */
	public static Vector power(Vector a, double pow) {
		Vector res = new Vector(a.length);
		if (pow == 0.5) {
			for (int i = 0; i < a.length; i++) {
				res.v[i] = Math.sqrt(a.v[i]);
			}
		} else {
			for (int i = 0; i < a.length; i++) {
				res.v[i] = Math.pow(a.v[i], pow);
			}

		}
		return res;
	}

	/**
	 * Multiplie ce vecteur par un scalaire scalar
	 * @param scalar scalaire à utiliser
	 * @return this
	 */
	public Vector scale(double scalar) {
		for (int i = 0; i < length; i++) {
			v[i] *= scalar;
		}
		return this;
	}

	/**
	 * Multiplie ce vecteur par un autre vecteur, élement à élement
	 * @param v vecteur à multiplier
	 * @return
	 */
	public Vector scale(Vector v) {
		if (length != v.length) {
			throw new RuntimeException("Incompatible shape : (" + length + ") and (" + v.length + ")");
		}
		for (int i = 0; i < length; i++) {
			this.v[i] *= v.v[i];
		}
		return this;
	}

	/**
	 * Multiplie un vecteur par un scalaire
	 * @param a vecteur à multiplier
	 * @param scalar scalaire à utiliser
	 * @return a_i = a_i * scalar
	 */
	public static Vector scale(Vector a, double scalar) {
		Vector res = new Vector(a.length);
		for (int i = 0; i < a.length; i++) {
			res.v[i] = a.v[i] * scalar;
		}
		return res;
	}

	/**
	 * Multiplie deux vecteurs ensemble, élement à élement
	 * @param a premier vecteur à multiplier
	 * @param b second vecteur à multiplier
	 * @return r_i = a_i * b_i
	 */
	public static Vector scale(Vector a, Vector b) {
		if (a.length != b.length) {
			throw new RuntimeException("Incompatible shape : (" + a.length + ") and (" + b.length + ")");
		}
		Vector res = new Vector(a.length);
		for (int i = 0; i < a.length; i++) {
			res.v[i] = a.v[i] * b.v[i];
		}
		return res;
	}
	
	/**
	 * Renvoie la différence a-b
	 */
	public static Vector sub(Vector a, Vector b) {
		if(a.length != b.length) {
			throw new RuntimeException("Incompatible shape : (" + a.length +") and (" + b.length + ")");
		}
		Vector c = new Vector(a.length);
		for(int i = 0 ; i < a.length ; i++) {
			c.v[i] = a.v[i]-b.v[i];
		}
		return c;
	}

	/**
	 * Renvoie la somme de ce vecteur
	 */
	public double sum() {
		double sum = 0;
		for (int i = 0; i < v.length; i++) {
			sum += v[i];
		}
		return sum;
	}

	/**
	 * Transforme ce vecteur en matrice colonne
	 */
	public Matrix to_column_matrix() {
		Matrix m = new Matrix(1, this.length);
		m.set_column(0, this);
		return m;
	}

	/**
	 * Transforme ce vecteur en matrice ligne
	 */
	public Matrix to_row_matrix() {
		Matrix m = new Matrix(this.length, 1);
		m.set_row(0, this);
		return m;
	}

	/**
	 * Permet d'afficher ce vecteur sous la forme (elem1, elem2, elem3, .. elemn)
	 */
	@Override
	public String toString() {
		StringBuffer sb = new StringBuffer();
		sb.append('(');
		for (int i = 0; i < this.length - 1; i++) {
			sb.append(this.v[i]);
			sb.append(", ");
		}
		sb.append(this.v[this.length - 1]);
		sb.append(')');
		return sb.toString();
	}

	public int[] to_int_array() {
		int[] d = new int[this.length];
		for(int i = 0 ; i < this.length ; i++) {
			d[i] = (int)Math.round(v[i]);
		}
		return d;
	}

	public static Vector random_gaussian_vector(int nb_params) {
		Vector v = new Vector(nb_params);
		for(int i = 0 ; i < nb_params ; i++) {
			v.v[i] = RandomGenerator.gaussian(1);
		}
		return v;
	}

	public Vector set_len(double len) {
		double norm = this.norm();
		return scale(len/norm);
	}

	private double norm() {
		double s = 0;
		for(int i = 0 ; i < length ; i++) {
			s += v[i]*v[i];
		}
		return Math.sqrt(s);
	}
}
