package math;
/**
 * Matrice � utiliser pour l'optimisation RMSProp, elle contient la d�riv�e et l'acceleration
 */
public class TrainableMatrix extends Matrix {
	public Matrix grad;

	public TrainableMatrix(int width, int height) {
		super(width, height);
		grad = new Matrix(width, height);
	}
}
