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
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

import javax.imageio.ImageIO;

public class Matrix {
	public double[][] v;
	public int width;
	public int height;
	
	public static final int AXIS_HEIGHT = 0;
	public static final int AXIS_WIDTH = 1;
	
	
	public Matrix(int width, int height) {
		v = new double[height][width];
		this.width = width;
		this.height = height;
	}
	
	public Matrix(Matrix x) {
		width = x.width;
		height = x.height;
		v = new double[height][width];
		for(int i = 0 ; i < height ;i++) {
			for(int j = 0 ; j < width ; j++) {
				v[i][j] = x.v[i][j];
			}
		}
	}
	
	public Matrix mult(Matrix b) {
		return Matrix.mult(this, b);
	}
	
	public static Matrix mult(Matrix a, Matrix b) {
		
		if(b.height != a.width) {
			throw new RuntimeException("Incompatible shape with ("+a.width+", "+a.height+") and ("+b.width+", "+b.height+") "+a.width+" != "+b.height);
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
	
	public Matrix mult_transposed(Matrix b) {
		return Matrix.mult_transposed(this, b);
	}
	
	public static Matrix mult_transposed(Matrix a, Matrix b) {
		if(b.height != a.width) {
			throw new RuntimeException("Incompatible shape with ("+a.width+", "+a.height+") and ("+b.width+", "+b.height+") "+a.width+" != "+b.height);
		}
		Matrix res = new Matrix(a.height, b.width);
		
		double sum;
		for(int j = 0 ; j < a.height ; j++) {
			for(int i = 0 ; i < b.width; i++) {
				sum = 0;
				for(int k = 0 ; k < a.width ; k++) {
					sum += a.v[j][k]*b.v[k][i];
				}
				res.v[i][j] = sum;
			}
		}
		return res;
	}
	
	public static double l2norm(Matrix a) {
		double n = 0;
		for(double[] b : a.v) {
			for(double c : b) {
				n += c*c;
			}
		}
		return Math.sqrt(n);
	}
	
	public double l2norm() {
		return Matrix.l2norm(this);
	}
	
	public Matrix parralel_mult(Matrix b) {
		return Matrix.parralel_mult(this, b);
	}
	
	public static Matrix parralel_mult(Matrix A, Matrix B) {
		if(B.height != A.width) {
			throw new RuntimeException("Incompatible shape with "+A.shape()+" and "+B.shape()+" "+A.width+" != "+B.height);
		}
		//System.out.println("Multiplying"+A.shape()+" "+B.shape());
		int threadNumber = Runtime.getRuntime().availableProcessors();
		while(A.height%threadNumber != 0)
			threadNumber--;
		Matrix C = new Matrix(B.width, A.height);
		ExecutorService executor = Executors.newFixedThreadPool(threadNumber);
		List<Future<Matrix>> list = new ArrayList<Future<Matrix>>();

		int part = A.height / threadNumber;
		if (part < 1) {
			part = 1;
		}
		
		// System.out.println(A.height+" "+part);
		for (int i = 0; i < A.height ; i += part) {
			Callable<Matrix> worker = new LineMultiplier(A, B, i, i+part);
			Future<Matrix> submit = executor.submit(worker);
			list.add(submit);
		}

		// now retrieve the result
		int start = 0;
		Matrix CF;
		for (Future<Matrix> future : list) {
			try {
				CF = future.get();
				for (int i=start; i < start+part ; i += 1) {
					C.v[i] = CF.v[i-start];
				}
			} catch (InterruptedException e) {
				e.printStackTrace();
			} catch (ExecutionException e) {
				e.printStackTrace();
			}
			start+=part;
		}
		executor.shutdown();

		return C;
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
	
	public Vector get_column(int i) {
		if(i < 0 || i >= this.width) {
			throw new RuntimeException(i+" not valid column id");
		}
		Vector m = new Vector(this.height);
		for(int k = 0 ; k < this.height ; k++) {
			m.v[k] = this.v[k][i];
		}
		return m;
	} 
	
	public Vector get_row(int j) {
		if(j < 0 || j >= this.height) {
			throw new RuntimeException(j+" not valid row id");
		}
		Vector m = new Vector(this.width);
		for(int k = 0 ; k < this.width ; k++) {
			m.v[k] = this.v[j][k];
		}
		return m;
	} 
	
	public Matrix set_column(int i, Vector v) {
		if(i < 0 || i >= this.width) {
			throw new RuntimeException(i+" not valid column id");
		}
		if(v.length != this.height) {
			throw new RuntimeException("Assigning vector of length "+v.length+" to height "+this.height);
		}
		for(int k = 0 ; k < this.height ; k++) {
			this.v[k][i] = v.v[k];
		}
		return this;
	} 
	
	public Matrix set_row(int j, Vector v) {
		if(j < 0 || j >= this.height || v.length != this.width) {
			throw new RuntimeException(j+" not valid row id");
		}
		for(int k = 0 ; k < this.width ; k++) {
			this.v[j][k] = v.v[k];
		}
		return this;
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
	
	public Matrix hadamart_div(Matrix b) {
		return Matrix.hadamart_div(this, b);
	}
	
	public static Matrix hadamart_div(Matrix a, Matrix b) {
		if(b.height != a.height || b.width != a.width) {
			throw new RuntimeException("Incompatible shape with ("+a.width+", "+a.height+") and ("+b.width+", "+b.height+")");
		}
		Matrix res = new Matrix(a.width, a.height);
		
		for(int j = 0 ; j < res.height ; j++) {
			for(int i = 0 ; i < res.width ; i++) {
				res.v[j][i] = a.v[j][i]/b.v[j][i];
			}
		}
		return res;
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


	public Matrix add(Matrix res) {
		if(height != res.height || width != res.width) {
			throw new RuntimeException("Incompatible shape with ("+width+", "+height+") and ("+res.width+", "+res.height+")");
		}
		for(int j = 0 ; j < res.height ; j++) {
			for(int i = 0 ; i < res.width ; i++) {
				v[j][i] += res.v[j][i];
			}
		}
		return this;
	}
	
	public Matrix add(double value) {
		for(int i = 0 ; i < height ; i++) {
			for(int j = 0 ; j < width ; j++) {
				this.v[i][j] += value;
			}
		}
		return this;
	}
	
	public Matrix fill(double value) {
		for(int i = 0 ; i < height ; i++) {
			for(int j = 0 ; j < width ; j++) {
				this.v[i][j] = value;
			}
		}
		return this;
	}

	public Matrix add(Vector v, int axis) {
		if(axis == AXIS_HEIGHT) {
			if(v.length != this.width)
				throw new RuntimeException("Incorrect vector with shape "+v.length);
			for(int i = 0 ; i < this.height ; i++) {
				for(int j = 0 ; j < this.width ; j++) {
					this.v[i][j] += v.v[j];
				}
			}
			return this;
		} 
		if(axis == AXIS_WIDTH) {
			if(v.length != this.height)
				throw new RuntimeException("Incorrect vector with shape "+v.length);
			for(int i = 0 ; i < this.height ; i++) {
				double val = v.v[i];
				for(int j = 0 ; j < this.width ; j++) {
					this.v[i][j] += val;
				}
			}
			return this;
		}
		throw new RuntimeException("Incorrect axis : "+axis);
	}
	
	public Matrix T() {
		return Matrix.transpose(this);
	}
	
	public Matrix transpose() {
		return Matrix.transpose(this);
	}
	
	public void zero_pad(int pad) {
		width += 4;
		height += 4;
		double[][] new_v = new double[height][width];
		for(int i = 0 ; i < v.length ; i++) {
			for(int j = 0 ; j < v[i].length ; j++) {
				new_v[i+pad][j+pad] = v[i][j];
			}
		}
		v = new_v;
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
	
	public static Matrix scale(Matrix m, double scalar) {
		Matrix res = new Matrix(m.width, m.height);
		
		for(int j = 0 ; j < res.height ; j++) {
			for(int i = 0 ; i < res.width ; i++) {
				res.v[j][i] = m.v[j][i]*scalar;
			}
		}
		
		return res;
	}
	
	public Matrix scale(double scalar) {
		for(int j = 0 ; j < this.height ; j++) {
			for(int i = 0 ; i < this.width ; i++) {
				this.v[j][i] *= scalar;
			}
		}
		
		return this;
	}
	
	public Matrix scale(Vector v, int axis) {
		if(axis == AXIS_HEIGHT) {
			if(v.length != this.width)
				throw new RuntimeException("Incorrect vector with shape "+v.length);
			for(int i = 0 ; i < this.height ; i++) {
				for(int j = 0 ; j < this.width ; j++) {
					this.v[i][j] *= v.v[j];
				}
			}
			return this;
		} 
		if(axis == AXIS_WIDTH) {
			if(v.length != this.height)
				throw new RuntimeException("Incorrect vector with shape "+v.length);
			for(int i = 0 ; i < this.height ; i++) {
				double val = v.v[i];
				for(int j = 0 ; j < this.width ; j++) {
					this.v[i][j] *= val;
				}
			}
			return this;
		}
		throw new RuntimeException("Incorrect axis : "+axis);
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
	
	public Vector shape() {
		Vector v = new Vector(2);
		v.v[0] = this.width;
		v.v[1] = this.height;
		return v;
	}
	
	@Override
	public boolean equals(Object obj) {
		if(obj instanceof Matrix) {
			Matrix t = (Matrix)obj;
			if(t.width == this.width && t.height == this.height) {
				double epsilon = 1e-10;
				for(int i = 0 ; i < height ; i++) {
					for (int j = 0; j < width; j++) {
						if(Math.abs(t.v[i][j]-this.v[i][j]) > epsilon) {
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
	public BufferedImage visualize(String name, int dimension, boolean hasBiasOnZero, boolean write) {
		BufferedImage bf;
		final int scale = 4;
		bf = new BufferedImage(dimension*scale*height, dimension*scale, BufferedImage.TYPE_INT_ARGB);
		for(int i = 0 ; i < height ; i++) {
			double max = this.get_row(i).max();
			double min = this.get_row(i).min();
			for(int j = 0 ; j < dimension*scale ; j++) {
				for(int k = 0 ; k < dimension*scale ; k++) {
					int indice = dimension*(j/scale)+k/scale+((hasBiasOnZero)?1:0);
					
					int v = (int) (512*(this.v[i][indice]-min)/(max-min));
					v -= 256;
					int a=0, b=0;
					if(v < 0)
						a = -v;
					else
						b = v;
					
					a *= 1.5;
					b *= 1.5;
					if(a > 255)
						a = 255;
					if(b > 255)
						b = 255;
					int rgb = (0xFF << 24) + (0 << 8) + (a << 16)+b;
					bf.setRGB(i*dimension*scale+k, j, rgb);
				}
			}
		}
		if(write) {
			try {
				ImageIO.write(bf, "png", new File(name+".png"));
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		return bf;
	}
	
	public void print_values() {
		for(int i = 0 ; i < height ; i++) {
			for (int j = 0; j < width; j++) {
				double ds = v[i][j];
				System.out.print((int)(ds*1000)/1000.0f+"\t");
			}
			System.out.println();
		}
	}
	
	public String toString() {
		return "("+width+", "+height+")";
	}

	public Vector sum(int axis) {
		if(axis == AXIS_HEIGHT) {
			Vector v = new Vector(this.width);	
			
			for(int i = 0 ; i < this.width ; i++) {
				double sum = 0;
				for(int j = 0 ; j < this.height ; j++) {
					sum += this.v[j][i];
				}
				v.v[i] = sum;
			}
			return v;
		} 
		if(axis == AXIS_WIDTH) {
			Vector v = new Vector(this.height);
			for(int i = 0 ; i < this.height ; i++) {
				double sum = 0;
				for(int j = 0 ; j < this.width ; j++) {
					sum += this.v[i][j];
				}
				v.v[i] = sum;
			}
			return v;
		}
		throw new RuntimeException("Incorrect axis : "+axis);
	}

	public double sum() {
		double sum = 0;
		for(int i = 0 ; i < this.height ; i++) {
			for(int j = 0 ; j < this.width ; j++) {
				sum += this.v[i][j];
			}
		}
		return sum;
	}
}
