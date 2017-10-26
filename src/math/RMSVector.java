package math;

public class RMSVector extends Vector {
	public Vector grad;
	public Vector acc;
	public RMSVector(int length) {
		super(length);
		grad = new Vector(length);
		acc = new Vector(length);
	}

}
