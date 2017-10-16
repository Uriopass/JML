package math;

import java.util.concurrent.Callable;

public class LineMultiplier implements Callable<Matrix> {
	Matrix A;
	Matrix B;
	int start;
	int end;
	public Matrix C;

	public LineMultiplier(Matrix a, Matrix b, int s, int e) {
		A = a;
		B = b;
		C = new Matrix(b.width, e-s);
		start = s;
		end = e;
	}

	@Override
	public Matrix call() {
		for (int i = start; i < end; i++) {
			int dec = i-start;
			for (int k = 0; k < B.height; k++) {
				double u =  A.v[i][k];
				for(int j = 0;j < B.width;j++) {
					C.v[dec][j] += u * B.v[k][j];
				}
			}
		}
		return C;
	}
}