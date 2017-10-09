package math;

public class Matrix {
	public double[][] v;
	public int width;
	public int height;
	
	public Matrix(int width, int height) {
		v = new double[height][width];
		this.width = width;
		this.height = height;
	}
	
	public Matrix mult(Matrix b) {
		return Matrix.mult(this, b);
	}
	
	public static Matrix mult(Matrix a, Matrix b) {
		if(b.height != a.width) {
			throw new RuntimeException("Incompatible shape with ("+a.width+", "+a.height+") and ("+b.width+", "+b.height+")");
		}
		Matrix res = new Matrix(b.width, a.height);
		
		double sum;
		for(int j = 0 ; j < res.height ; j++) {
			for(int i = 0 ; i < res.width ; i++) {
				sum = 0;
				for(int k = 0 ; k < a.width ; k++) {
					sum += a.v[j][k]*b.v[k][i];
				}
				res.v[j][i] = sum;
			}
		}
		return res;
	}
	
	public Vector dot(Vector v) {
		return Matrix.dot(this, v);
	}
	
	public static Vector dot(Matrix m, Vector v) {
		if(m.width != v.length) {
			throw new RuntimeException("Incompatible shape with ("+m.width+", "+m.height+") and (1, "+v.length+")");
		}
		Vector res = new Vector(m.height);
		double sum;
		for(int k = 0 ; k < m.height ; k++) {
			sum = 0;
			for(int i = 0 ; i < m.width ; i++) {
				sum += m.v[k][i]*v.v[i];
			}
			res.v[k] = sum;
		}
		return res;
	}
	
	public Vector getColumn(int i) {
		if(i < 0 || i >= this.width) {
			throw new RuntimeException(i+" not valid column id");
		}
		Vector m = new Vector(this.height);
		for(int k = 0 ; k < this.height ; k++) {
			m.v[k] = this.v[k][i];
		}
		return m;
	} 
	
	public Vector getRow(int j) {
		if(j < 0 || j >= this.height) {
			throw new RuntimeException(j+" not valid row id");
		}
		Vector m = new Vector(this.width);
		for(int k = 0 ; k < this.width ; k++) {
			m.v[k] = this.v[j][k];
		}
		return m;
	} 
	
	public Matrix hadamart(Matrix b) {
		return Matrix.hadamart(this, b);
	}
	
	public static Matrix hadamart(Matrix a, Matrix b) {
		if(b.height != a.height || b.width != a.width) {
			throw new RuntimeException("Incompatible shape with ("+a.width+", "+a.height+") and ("+b.width+", "+b.height+")");
		}
		Matrix res = new Matrix(a.width, a.height);
		
		for(int j = 0 ; j < res.height ; j++) {
			for(int i = 0 ; i < res.width ; i++) {
				res.v[j][i] = a.v[j][i]*b.v[j][i];
			}
		}
		return res;
	}
	
	public Matrix add(Matrix b) {
		return Matrix.add(this, b);
	}
	
	public static Matrix add(Matrix a, Matrix b) {
		if(b.height != a.height || b.width != a.width) {
			throw new RuntimeException("Incompatible shape with ("+a.width+", "+a.height+") and ("+b.width+", "+b.height+")");
		}
		Matrix res = new Matrix(a.width, a.height);
		
		for(int j = 0 ; j < res.height ; j++) {
			for(int i = 0 ; i < res.width ; i++) {
				res.v[j][i] = a.v[j][i]+b.v[j][i];
			}
		}
		
		return res;
	}
	
	public Matrix transpose() {
		return Matrix.transpose(this);
	}
	
	public static Matrix transpose(Matrix m) {
		Matrix res = new Matrix(m.height, m.width);
		
		for(int j = 0 ; j < res.height ; j++) {
			for(int i = 0 ; i < res.width ; i++) {
				res.v[j][i] = m.v[i][j];
			}
		}
		
		return res;
	}
	
	public Matrix scale(double scalar) {
		return Matrix.scale(this, scalar);
	}
	
	public static Matrix scale(Matrix m, double scalar) {
		Matrix res = new Matrix(m.width, m.height);
		
		for(int j = 0 ; j < res.height ; j++) {
			for(int i = 0 ; i < res.width ; i++) {
				res.v[j][i] = m.v[j][i]*scalar;
			}
		}
		
		return res;
	}
	
	public Matrix scaleInPlace(double scalar) {
		for(int j = 0 ; j < this.height ; j++) {
			for(int i = 0 ; i < this.width ; i++) {
				this.v[j][i] *= scalar;
			}
		}
		
		return this;
	}

	public double max() {
		return Matrix.max(this);
	}
	
	public static double max(Matrix m) {
		double max = m.v[0][0];
		for(int j = 0 ; j < m.height ; j++) {
			for(int i = 0 ; i < m.width ; i++) {
				max = Math.max(max, m.v[j][i]);
			}
		}
		
		return max;
	}
	
	public double min() {
		return Matrix.min(this);
	}
	
	public static double min(Matrix m) {
		double min = m.v[0][0];
		for(int j = 0 ; j < m.height ; j++) {
			for(int i = 0 ; i < m.width ; i++) {
				min = Math.min(min, m.v[j][i]);
			}
		}
		
		return min;
	}

	public void addInPlace(Matrix res) {
		if(height != res.height || width != res.width) {
			throw new RuntimeException("Incompatible shape with ("+width+", "+height+") and ("+res.width+", "+res.height+")");
		}
		for(int j = 0 ; j < res.height ; j++) {
			for(int i = 0 ; i < res.width ; i++) {
				v[j][i] = v[j][i]+res.v[j][i];
			}
		}
	}
	
	public String toString() {
		return "("+width+", "+height+")";
		
	}
}
