package math;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import javax.imageio.ImageIO;

/**
 * Classe principale de calcul de matrice et de gestion de matrices, Matrix est
 * une surcouche au dessus d'un double tableau de double.
 */
public class Matrix {
	// Axe des lignes, utilisé pour effectuer une somme sur un seul axe par exemple
	public static final int AXIS_HEIGHT = 0;
	// Axe des colonnes
	public static final int AXIS_WIDTH = 1;

	// Valeurs de la matrices
	public double[][] v;

	// Nombres de colonnes
	public int width;

	// Nombre de lignes
	public int height;

	/**
	 * Crée une matrice remplie de zéros
	 * 
	 * @param width
	 *            nombre de colonnes
	 * @param height
	 *            nombre de lignes
	 */
	public Matrix(int width, int height) {
		v = new double[height][width];
		this.width = width;
		this.height = height;
	}

	/**
	 * Crée une copie de la matrice passée en argument
	 * 
	 * @param x
	 *            Matrice à copier
	 */
	public Matrix(Matrix x) {
		if(x == null) {
			v = new double[0][0];
			return;
		}
		width = x.width;
		height = x.height;
		v = new double[height][width];
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				v[i][j] = x.v[i][j];
			}
		}
	}

	/**
	 * Effectue une addition de deux matrices, élément-à-élement.
	 * 
	 * @param a
	 *            Première matrice à additionner
	 * @param b
	 *            Deuxième matrice à additionner
	 * @return nouvelle matrice contenant la somme
	 */
	public static Matrix add(Matrix a, Matrix b) {
		if (b.height != a.height || b.width != a.width) {
			throw new RuntimeException("Incompatible shape with (" + a.width + ", " + a.height + ") and (" + b.width
					+ ", " + b.height + ")");
		}
		Matrix res = new Matrix(a.width, a.height);

		for (int j = 0; j < res.height; j++) {
			for (int i = 0; i < res.width; i++) {
				res.v[j][i] = a.v[j][i] + b.v[j][i];
			}
		}

		return res;
	}

	/**
	 * Ajoute à tout les élements de cette matrice un scalaire
	 * 
	 * @param value
	 *            scalaire à ajouter
	 * @return this
	 */
	public Matrix add(double value) {
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				this.v[i][j] += value;
			}
		}
		return this;
	}

	/**
	 * Additionne une matrice à cette matrice, élément-à-élement
	 * 
	 * @param b
	 *            Matrice à additionner
	 * @return this
	 */
	public Matrix add(Matrix b) {
		if (height != b.height || width != b.width) {
			throw new RuntimeException(
					"Incompatible shape with (" + width + ", " + height + ") and (" + b.width + ", " + b.height + ")");
		}
		for (int j = 0; j < b.height; j++) {
			for (int i = 0; i < b.width; i++) {
				v[j][i] += b.v[j][i];
			}
		}
		return this;
	}

	/**
	 * Ajoute un vecteur le long d'un axe
	 * 
	 * @param v
	 *            vecteur à ajouter
	 * @param axis
	 *            axe utilisé pour la somme
	 * @return this
	 */
	public Matrix add(Vector v, int axis) {
		if (axis == AXIS_HEIGHT) {
			if (v.length != this.width)
				throw new RuntimeException("Incorrect vector with shape " + v.length);
			for (int i = 0; i < this.height; i++) {
				for (int j = 0; j < this.width; j++) {
					this.v[i][j] += v.v[j];
				}
			}
			return this;
		}
		if (axis == AXIS_WIDTH) {
			if (v.length != this.height)
				throw new RuntimeException("Incorrect vector with shape " + v.length);
			for (int i = 0; i < this.height; i++) {
				double val = v.v[i];
				for (int j = 0; j < this.width; j++) {
					this.v[i][j] += val;
				}
			}
			return this;
		}
		throw new RuntimeException("Incorrect axis : " + axis);
	}

	/**
	 * Effectue argmax le long d'un axe
	 * 
	 * @param axis
	 *            axe à utiliser
	 * @return vecteur contenant les résultats
	 */
	public Vector argmax(int axis) {
		if (axis == Matrix.AXIS_HEIGHT) {
			Vector v = new Vector(this.width);
			for (int i = 0; i < this.width; i++) {
				v.v[i] = this.get_column(i).argmax();
			}
			return v;
		}
		if (axis == Matrix.AXIS_WIDTH) {
			Vector v = new Vector(this.height);
			for (int i = 0; i < this.height; i++) {
				v.v[i] = this.get_row(i).argmax();
			}
			return v;
		}
		return null;
	}

	/**
	 * Effectue le produit scalaire entre une matrice et un vecteur
	 * 
	 * @param v
	 *            le vecteur à multiplier
	 * @return this . v
	 */
	public Vector dot(Vector v) {
		return Matrix.dot(this, v);
	}

	public static Vector dot(Matrix m, Vector v) {
		if (m.width != v.length) {
			throw new RuntimeException(
					"Incompatible shape with (" + m.width + ", " + m.height + ") and (1, " + v.length + ")");
		}
		Vector res = new Vector(m.height);
		double sum;
		for (int k = 0; k < m.height; k++) {
			sum = 0;
			for (int i = 0; i < m.width; i++) {
				sum += m.v[k][i] * v.v[i];
			}
			res.v[k] = sum;
		}
		return res;
	}

	/**
	 * Vérifie si cette matrice est égal à une autre matrice avec une précision
	 * 1e-10
	 */
	@Override
	public boolean equals(Object obj) {
		if (obj instanceof Matrix) {
			Matrix t = (Matrix) obj;
			if (t.width == this.width && t.height == this.height) {
				double epsilon = 1e-10;
				for (int i = 0; i < height; i++) {
					for (int j = 0; j < width; j++) {
						if (Math.abs(t.v[i][j] - this.v[i][j]) > epsilon) {
							return false;
						}
					}
				}
				return true;
			}
			return false;
		}
		return false;
	}

	/**
	 * Remplie cette matrice avec un scalaire
	 * 
	 * @param value
	 *            scalaire utilisé pour remplir
	 * @return this
	 */
	public Matrix fill(double value) {
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				this.v[i][j] = value;
			}
		}
		return this;
	}

	/**
	 * Permet de récuperer une colonne de la matrice sous la forme d'un vecteur
	 * 
	 * @param i
	 *            indice de la colonne
	 */
	public Vector get_column(int i) {
		if (i < 0 || i >= this.width) {
			throw new RuntimeException(i + " not valid column id");
		}
		Vector m = new Vector(this.height);
		for (int k = 0; k < this.height; k++) {
			m.v[k] = this.v[k][i];
		}
		return m;
	}

	/**
	 * Permet de récuperer une ligne de la matrice sous la forme d'un vecteur
	 * 
	 * @param j
	 *            l'indice de la ligne
	 */
	public Vector get_row(int j) {
		if (j < 0 || j >= this.height) {
			throw new RuntimeException(j + " not valid row id");
		}
		Vector m = new Vector(this.width);
		for (int k = 0; k < this.width; k++) {
			m.v[k] = this.v[j][k];
		}
		return m;
	}

	/**
	 * Effectue une multiplication élément-à-élement de de la matrice actuelle avec
	 * la nouvelle matrice La multiplication se fait en-place, une nouvelle matrice
	 * n'est donc pas allouée
	 * 
	 * @param b
	 *            matrice à multiplier
	 * @return this
	 */
	public Matrix hadamart(Matrix b) {
		if (b.height != this.height || b.width != this.width) {
			throw new RuntimeException("Incompatible shape with (" + this.width + ", " + this.height + ") and ("
					+ b.width + ", " + b.height + ")");
		}
		for (int j = 0; j < height; j++) {
			for (int i = 0; i < width; i++) {
				v[j][i] *= b.v[j][i];
			}
		}
		return this;
	}

	/**
	 * Effectue une multiplication élément-à-élement de deux matrices
	 * 
	 * @param a
	 *            Première matrice à multiplier
	 * @param b
	 *            Deuxième matrice à multiplier
	 * @return nouvelle matrice contenant le produit
	 */
	public static Matrix hadamart(Matrix a, Matrix b) {
		if (b.height != a.height || b.width != a.width) {
			throw new RuntimeException("Incompatible shape with (" + a.width + ", " + a.height + ") and (" + b.width
					+ ", " + b.height + ")");
		}
		Matrix res = new Matrix(a.width, a.height);

		for (int j = 0; j < res.height; j++) {
			for (int i = 0; i < res.width; i++) {
				res.v[j][i] = a.v[j][i] * b.v[j][i];
			}
		}
		return res;
	}

	/**
	 * Crée la matrice identité
	 * 
	 * @param size
	 *            taille de la matrice
	 * @return
	 */
	public static Matrix identity(int size) {
		Matrix id = new Matrix(size, size);
		for (int i = 0; i < size; i++) {
			id.v[i][i] = 1;
		}
		return id;
	}

	/**
	 * Renvoie la norme L2 de cette matrice, définie par sqrt(somme(x_ij*x_ij))
	 */
	public double l2norm() {
		double n = 0;
		for (double[] b : v) {
			for (double c : b) {
				n += c * c;
			}
		}
		return Math.sqrt(n);
	}

	/**
	 * Renvoie le maximum de cette matrice
	 * 
	 * @return le maximum
	 */
	public double max() {
		double max = this.v[0][0];
		for (int j = 0; j < this.height; j++) {
			for (int i = 0; i < this.width; i++) {
				max = Math.max(max, this.v[j][i]);
			}
		}

		return max;
	}

	/**
	 * Renvoie le minimum de cette matrice
	 * 
	 * @return le minimum
	 */
	public double min() {
		double min = this.v[0][0];
		for (int j = 0; j < this.height; j++) {
			for (int i = 0; i < this.width; i++) {
				min = Math.min(min, this.v[j][i]);
			}
		}

		return min;
	}

	/**
	 * Multiplie cette matrice par la matrice b
	 * 
	 * @param b
	 *            la matrice à multiplier
	 * @return this * b
	 */
	public Matrix mult(Matrix b) {
		return Matrix.mult(this, b);
	}

	public static Matrix mult(Matrix a, Matrix b) {
		if (b.height != a.width) {
			throw new RuntimeException("Incompatible shape with (" + a.width + ", " + a.height + ") and (" + b.width
					+ ", " + b.height + ") " + a.width + " != " + b.height);
		}
		Matrix res = new Matrix(b.width, a.height);
		double sum;
		for (int j = 0; j < res.height; j++) {
			for (int i = 0; i < res.width; i++) {
				sum = 0;
				for (int k = 0; k < a.width; k++) {
					sum += a.v[j][k] * b.v[k][i];
				}
				res.v[j][i] = sum;
			}
		}
		return res;
	}

	/**
	 * Idem que mult, mais utilisant le maximum de processeurs disponibles
	 * 
	 * @param b
	 *            la matrice à multiplier
	 * @return this * b
	 */
	public Matrix parralel_mult(Matrix b) {
		return Matrix.parralel_mult(this, b);
	}

	public static Matrix parralel_mult(Matrix A, Matrix B) {
		if (B.height != A.width) {
			throw new RuntimeException(
					"Incompatible shape with " + A.shape() + " and " + B.shape() + " " + A.width + " != " + B.height);
		}

		int threadNumber = Runtime.getRuntime().availableProcessors();
		while (A.height % threadNumber != 0)
			threadNumber--;
		if (threadNumber == 1) {
			return A.mult(B);
		}
		int part = A.height / threadNumber;
		if (part < 1) {
			part = 1;
		}
		Matrix C = new Matrix(B.width, A.height);
		ExecutorService executor = Executors.newFixedThreadPool(threadNumber);
		List<Future<Matrix>> list = new ArrayList<Future<Matrix>>();

		for (int i = 0; i < A.height; i += part) {
			Callable<Matrix> worker = new LineMultiplier(A, B, i, i + part);
			Future<Matrix> submit = executor.submit(worker);
			list.add(submit);
		}

		int start = 0;
		Matrix CF;
		for (Future<Matrix> future : list) {
			try {
				CF = future.get();
				for (int i = start; i < start + part; i += 1) {
					C.v[i] = CF.v[i - start];
				}
			} catch (InterruptedException e) {
				e.printStackTrace();
			} catch (ExecutionException e) {
				e.printStackTrace();
			}
			start += part;
		}
		executor.shutdown();
		return C;
	}

	/**
	 * Affiche les valeurs de cette matrice, avec une précision de 3 chiffres
	 */
	public void print_values() {
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				double ds = v[i][j];
				System.out.print((int) (ds * 1000) / 1000.0f + "\t");
			}
			System.out.println();
		}
	}

	/**
	 * Multiplie tout les élements de cette matrice par un scalaire
	 * 
	 * @param value
	 *            scalaire à multiplier
	 * @return this
	 */
	public Matrix scale(double scalar) {
		for (int j = 0; j < this.height; j++) {
			for (int i = 0; i < this.width; i++) {
				this.v[j][i] *= scalar;
			}
		}
		return this;
	}

	/**
	 * Multiplie un vecteur le long d'un axe
	 * 
	 * @param v
	 *            vecteur à utiliser
	 * @param axis
	 *            axe à utiliser
	 * @return this
	 */
	public Matrix scale(Vector v, int axis) {
		if (axis == AXIS_HEIGHT) {
			if (v.length != this.width)
				throw new RuntimeException("Incorrect vector with shape " + v.length);
			for (int i = 0; i < this.height; i++) {
				for (int j = 0; j < this.width; j++) {
					this.v[i][j] *= v.v[j];
				}
			}
			return this;
		}
		if (axis == AXIS_WIDTH) {
			if (v.length != this.height)
				throw new RuntimeException("Incorrect vector with shape " + v.length);
			for (int i = 0; i < this.height; i++) {
				double val = v.v[i];
				for (int j = 0; j < this.width; j++) {
					this.v[i][j] *= val;
				}
			}
			return this;
		}
		throw new RuntimeException("Incorrect axis : " + axis);
	}

	/**
	 * Multiplie tout les élements d'une matrice par un scalaire
	 * 
	 * @param value
	 *            scalaire utilisé
	 * @param m
	 *            Matrice à multiplier
	 * @return nouvelle matrice contenant le résultat
	 */
	public static Matrix scale(Matrix m, double scalar) {
		Matrix res = new Matrix(m.width, m.height);

		for (int j = 0; j < res.height; j++) {
			for (int i = 0; i < res.width; i++) {
				res.v[j][i] = m.v[j][i] * scalar;
			}
		}

		return res;
	}

	/**
	 * Permet de remplacer une colonne de la matrice par un nouveau vecteur
	 * 
	 * @param i
	 *            indice de la colonne
	 * @param v
	 *            vecteur à utiliser
	 * @return this
	 */
	public Matrix set_column(int i, Vector v) {
		if (i < 0 || i >= this.width) {
			throw new RuntimeException(i + " not valid column id");
		}
		if (v.length != this.height) {
			throw new RuntimeException("Assigning vector of length " + v.length + " to height " + this.height);
		}
		for (int k = 0; k < this.height; k++) {
			this.v[k][i] = v.v[k];
		}
		return this;
	}

	/**
	 * Permet de remplacer une ligne de la matrice par un nouveau vecteur
	 * 
	 * @param j
	 *            indice de la ligne
	 * @param v
	 *            vecteur à utiliser
	 * @return this
	 */
	public Matrix set_row(int j, Vector v) {
		if (j < 0 || j >= this.height || v.length != this.width) {
			throw new RuntimeException(j + " not valid row id");
		}
		for (int k = 0; k < this.width; k++) {
			this.v[j][k] = v.v[k];
		}
		return this;
	}

	/**
	 * Renvoie la forme de cette matrice sous la forme d'un vecteur (c, l) avec c le
	 * nombre de colonnes et l le nombre de lignes
	 * 
	 * @return un vecteur de dimension 2
	 */
	public Vector shape() {
		Vector v = new Vector(2);
		v.v[0] = this.width;
		v.v[1] = this.height;
		return v;
	}

	/**
	 * Effectue la somme de cette matrice
	 */
	public double sum() {
		double sum = 0;
		for (int i = 0; i < this.height; i++) {
			for (int j = 0; j < this.width; j++) {
				sum += this.v[i][j];
			}
		}
		return sum;
	}

	/**
	 * Effectue la somme de cette matrice le long d'un axe
	 * 
	 * @param axis
	 *            axe à utiliser
	 * @return vecteur contenant les résultats de chaque somme
	 */
	public Vector sum(int axis) {
		if (axis == AXIS_HEIGHT) {
			Vector v = new Vector(this.width);

			for (int i = 0; i < this.width; i++) {
				double sum = 0;
				for (int j = 0; j < this.height; j++) {
					sum += this.v[j][i];
				}
				v.v[i] = sum;
			}
			return v;
		}
		if (axis == AXIS_WIDTH) {
			Vector v = new Vector(this.height);
			for (int i = 0; i < this.height; i++) {
				double sum = 0;
				for (int j = 0; j < this.width; j++) {
					sum += this.v[i][j];
				}
				v.v[i] = sum;
			}
			return v;
		}
		throw new RuntimeException("Incorrect axis : " + axis);
	}

	/**
	 * Renvoie la transposée de la matrice
	 * 
	 * @return transposée de la matrice
	 */
	public Matrix T() {
		return Matrix.transpose(this);
	}

	/**
	 * Renvoie la transposée de la matrice
	 * 
	 * @return transposée de la matrice
	 */
	public Matrix transpose() {
		return Matrix.transpose(this);
	}

	/**
	 * Renvoie la transposée d'une matrice m
	 * 
	 * @param m
	 *            Matrice à transposer
	 * @return transposée de la matrice
	 */
	public static Matrix transpose(Matrix m) {
		Matrix res = new Matrix(m.height, m.width);

		for (int j = 0; j < res.height; j++) {
			for (int i = 0; i < res.width; i++) {
				res.v[j][i] = m.v[i][j];
			}
		}

		return res;
	}

	/**
	 * Affiche les dimensions de cette matrice
	 */
	public String toString() {
		return "(" + width + ", " + height + ")";
	}

	/**
	 * Permet de visualiser cette matrice sous la forme d'une image, en supposant
	 * que chaque élément en hauteur est un filtre, et en mettant ces filtres les
	 * uns à côté des autres.
	 * 
	 * @param name
	 *            nom du fichier à utiliser pour l'écriture
	 * @param dimension
	 *            dimension de l'image (souvent 28)
	 * @param f_w
	 *            nombre de filtres en largeur
	 * @param f_h
	 *            nombre de filtres en hauteur
	 * @param write
	 *            écrit ou non l'image générée
	 * @param black_and_white 
	 * 			  genere l'image and utilisant rouge et bleu ou noir et blanc
	 * @return l'image générée par la fonction
	 */
	public BufferedImage visualize(String name, int dimension, int f_w, int f_h, boolean write, boolean black_and_white) {
		if (f_w * f_h != height) {
			throw new RuntimeException("Dimensions don't match height " + f_w * f_h + " != " + height);
		}
		BufferedImage bf;
		final int scale = 4;
		bf = new BufferedImage(dimension * scale * f_w, dimension * scale * f_h, BufferedImage.TYPE_INT_ARGB);
		for (int i = 0; i < f_h; i++) {
			for (int i2 = 0; i2 < f_w; i2++) {
				int im_indice = i * f_w + i2;
				double max = this.get_row(im_indice).max();
				double min = this.get_row(im_indice).min();
				for (int j = 0; j < dimension * scale; j++) {
					for (int k = 0; k < dimension * scale; k++) {
						int indice = dimension * (j / scale) + k / scale;
						int rgb;
						if(!black_and_white) {
							int v = (int) (512 * (this.v[im_indice][indice] - min) / (max - min));
							v -= 256;
							int a = 0, b = 0;
							if (v < 0)
								a = -v;
							else
								b = v;
	
							a *= 1.5;
							b *= 1.5;
							if (a > 255)
								a = 255;
							if (b > 255)
								b = 255;
							rgb = (0xFF << 24) + (0 << 8) + (a << 16) + b;
						} else {
							int v = (int) (255 * (this.v[im_indice][indice] - min) / (max - min));
							rgb = (0xFF << 24) + (v << 8) + (v << 16) + v;
						}
						bf.setRGB(i2 * dimension * scale + k, i * dimension * scale + j, rgb);
					}
				}
			}
		}
		if (write) {
			try {
				ImageIO.write(bf, "png", new File(name + ".png"));
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		return bf;
	}

	/**
	 * pad une matrice avec des zéros
	 * 
	 * @param pad
	 *            taille du pad
	 */
	public void zero_pad(int pad) {
		width += 2 * pad;
		height += 2 * pad;
		double[][] new_v = new double[height][width];
		for (int i = 0; i < v.length; i++) {
			for (int j = 0; j < v[i].length; j++) {
				new_v[i + pad][j + pad] = v[i][j];
			}
		}
		v = new_v;
	}
}
