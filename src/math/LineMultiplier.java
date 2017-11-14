package math;

import java.util.concurrent.Callable;

/**
 * Utilisé pour la multiplication parallèle de matrices, cette classe effectue une partie des multiplications définie par start et end
 */
public class LineMultiplier implements Callable<Matrix> {
	Matrix A;
	Matrix B;
	int start;
	int end;
	public Matrix C;

	public LineMultiplier(Matrix a, Matrix b, int start, int end) {
		A = a;
		B = b;
		C = new Matrix(b.width, end - start);
		this.start = start;
		this.end = end;
	}

	@Override
	public Matrix call() {
		for (int i = start; i < end; i++) {
			int dec = i - start;
			for (int k = 0; k < B.height; k++) {
				double u = A.v[i][k];
				for (int j = 0; j < B.width; j++) {
					C.v[dec][j] += u * B.v[k][j];
				}
			}
		}
		return C;
	}
}