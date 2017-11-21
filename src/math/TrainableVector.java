package math;
/**
 * Vecteur � utiliser pour l'optimisation RMSProp, elle contient la d�riv�e et l'acceleration
 */
public class TrainableVector extends Vector {
	public Vector grad;
	public Vector acc;

	public TrainableVector(int length) {
		super(length);
		grad = new Vector(length);
		acc = new Vector(length);
	}

}
