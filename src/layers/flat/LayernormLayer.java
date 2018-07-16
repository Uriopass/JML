package layers.flat;

import java.io.FileNotFoundException;
import java.util.Collection;
import java.util.HashMap;

import layers.FlatLayer;
import layers.Parameters;
import layers.TrainableVectors;
import math.Matrix;
import math.TrainableVector;
import math.Vector;

public class LayernormLayer implements FlatLayer, TrainableVectors {

	final static double epsilon = 1e-4;

	private Matrix xmu, carre;

	private Vector var, sqrtvar, invvar, mu;

	public HashMap<String, TrainableVector> vectors = new HashMap<>();

	public LayernormLayer(Parameters param) {
		vectors.put("gamma", new TrainableVector(1));
		vectors.put("beta", new TrainableVector(1));
		
		vectors.get("gamma").fill(1);
	}

	@Override
	public Matrix forward(Matrix in, boolean training) {
		if(training || !training) {
			throw new RuntimeException("Not implemented yet");
		}
		// Step 1
		mu = in.sum(Matrix.AXIS_HEIGHT).scale(-1.0 / in.height);

		// Step 2
		in.add(mu, Matrix.AXIS_HEIGHT);
		xmu = new Matrix(in);

		// Step 3
		carre = Matrix.hadamart(xmu, xmu); // square
		
		// Step 4
		var = carre.sum(Matrix.AXIS_HEIGHT).scale(1.0 / in.height); // average

		// Step 5
		sqrtvar = new Vector(var).add(epsilon).power(0.5);

		// Step 6
		invvar = new Vector(sqrtvar).inverse();

		// Step 7
		in.scale(invvar, Matrix.AXIS_WIDTH);
		
		double gamma = vectors.get("gamma").v[0];
		double beta = vectors.get("beta").v[0];
		
		// Step 8
		in.scale(gamma);

		// Step 9
		in.add(beta);
		
		return in;
	}

	@Override
	public Matrix backward(Matrix dout, boolean train) {
		throw new RuntimeException("Not implemented yet");
		/*
		int N = dout.width;
		// Step 9
		Matrix dva3 = dout;
		if (train) {
			vectors.get("beta").grad.add(dout.sum(Matrix.AXIS_WIDTH).scale(1.0 / N));
			// Step 8
			vectors.get("gamma").grad.add(va2.hadamart(dva3).sum(Matrix.AXIS_WIDTH).scale(1.0 / N));
		}
		Matrix dva2 = dva3.scale(vectors.get("gamma"), Matrix.AXIS_WIDTH);
		// Step 7
		Vector dinvvar = new Matrix(xmu).hadamart(dva2).sum(Matrix.AXIS_WIDTH);
		Matrix dxmu = dva2.scale(invvar, Matrix.AXIS_WIDTH);
		// Step 6
		Vector dsqrtvar = sqrtvar.scale(sqrtvar).inverse().scale(-1).scale(dinvvar);
		// Step 5
		Vector dvar = var.add(epsilon).power(0.5).inverse().scale(dsqrtvar).scale(0.5);
		// Step 4
		Matrix dcarre = new Matrix(carre.width, carre.height);
		dcarre.fill(1).scale(1.0 / dout.width).scale(dvar, Matrix.AXIS_WIDTH);
		// Step 3
		dxmu.add(xmu.hadamart(dcarre).scale(2));
		// Step 2
		Vector dmu = dxmu.sum(Matrix.AXIS_WIDTH).scale(-1.0 / N);
		// Step 1
		dxmu.add(dmu, Matrix.AXIS_WIDTH);
		return dxmu;
		*/
	}
	
	public void write_to_file(String name) {
		vectors.get("gamma").to_row_matrix().write_to_file(name+"/gamma");
		vectors.get("beta").to_row_matrix().write_to_file(name+"/beta");
	}
	
	public void load_from_file(String name) throws FileNotFoundException {
		vectors.get("gamma").replace_by(Matrix.generate_from_file(name+"/gamma").get_row(0));
		vectors.get("beta").replace_by(Matrix.generate_from_file(name+"/beta").get_row(0));
	}
	
	@Override
	public String toString() {
		return "LayernormLayer()";
	}

	@Override
	public Collection<TrainableVector> get_trainable_vectors() {
		return vectors.values();
	}

}
