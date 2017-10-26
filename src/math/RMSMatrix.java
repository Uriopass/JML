package math;

public class RMSMatrix extends Matrix {
	public Matrix grad;
	public Matrix acc;
	
	
	public RMSMatrix(int width, int height) {
		super(width, height);
		grad = new Matrix(width, height);
		acc = new Matrix(width, height);
	}
}
