package perceptron;

import java.io.File;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;
import java.util.Scanner;

import math.Activations;
import math.Gradient;
import math.Matrix;
import math.Vector;

public class MultiLayerPerceptron {
	public int[] dims;
	public Matrix[] weights; // paramètres du modèle
	public Vector[] biases;

	public Matrix data;
	public int[] refs; // les réferences
	public double learning_rate = 0.001;
	public double learning_rate_decay = 0.794328235; // x^10 = 0.1
	public double reg = 0.0001;

	public Matrix activation_forward(Matrix x) {
		return Activations.TanH(x);
	}

	public Vector activation_forward(Vector x) {
		return Activations.TanH(x);
	}

	public double activation_backward(double x) {
		return Activations.TanH_backward(x);
	}

	public void weight_init(long seed) {
		Random r = new Random(seed);
		weights = new Matrix[dims.length - 1];
		biases = new Vector[weights.length];
		for (int k = 0; k < weights.length; k++) {
			weights[k] = new Matrix(dims[k], dims[k + 1]);
			biases[k] = new Vector(dims[k + 1]);
			double bound = 4 * Math.sqrt(6f / (dims[k] + dims[k + 1]));
			// System.out.println("mult:" + bound);
			for (int i = 0; i < weights[k].height; i++) {
				for (int j = 0; j < weights[k].width; j++) {
					weights[k].v[i][j] = bound * (r.nextDouble() * 2 - 1);
				}
			}
			for (int i = 0; i < biases[k].length; i++) {
				biases[k].v[i] = 0;// bound * (r.nextDouble() * 2 - 1);
			}
		}
	}

	public void write_weights(String name) {
		try {
			File f = new File(name + ".jml");
			if (!f.exists()) {
				f.createNewFile();
			}
			PrintWriter pw = new PrintWriter(f);
			pw.write("" + dims.length);
			pw.write('\n');
			for (int i = 0; i < dims.length; i++) {
				pw.write(dims[i] + " ");
			}
			pw.write('\n');

			for (int i = 0; i < weights.length; i++) {
				Matrix m = weights[i];
				for (int k = 0; k < m.height; k++) {
					for (int j = 0; j < m.width; j++) {
						pw.write(m.v[k][j] + " ");
					}
					pw.write('\n');
				}
				for (int k = 0; k < biases[i].length; k++) {
					pw.write(biases[i].v[k] + " ");
				}
				pw.write('\n');
			}
			pw.flush();
			pw.close();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	public void load_weights(String name) {
		try {
			File f = new File(name + ".jml");
			if (!f.exists()) {
				throw new RuntimeException("No file named" + name);
			}
			System.out.println("# Loading weights in "+name+".jml");
			Scanner sc = new Scanner(f);
			int dimslength = sc.nextInt();
			dims = new int[dimslength];
			for (int i = 0; i < dims.length; i++) {
				int dimsi = sc.nextInt();
				dims[i] = dimsi;
			}
			weights = new Matrix[dims.length - 1];
			biases = new Vector[dims.length - 1];
			for (int i = 0; i < dims.length - 1; i++) {
				weights[i] = new Matrix(dims[i], dims[i + 1]);
				biases[i] = new Vector(dims[i + 1]);
				for (int k = 0; k < weights[i].height; k++) {
					for (int j = 0; j < weights[i].width; j++) {
						//System.out.println(i+" "+k+" "+j+" "+sc.hasNextDouble());
						weights[i].v[k][j] = Double.parseDouble(sc.next());
					}
				}
				for (int k = 0; k < biases[i].length; k++) {
					biases[i].v[k] = Double.parseDouble(sc.next());
				}
			}
			sc.close();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	public int argmax(Vector v) {
		int max = 0;
		double maxV = v.v[0];
		for (int i = 1; i < v.length; i++) {
			if (v.v[i] > maxV) {
				maxV = v.v[i];
				max = i;
			}
		}
		return max;
	}

	public Matrix forward_layer(Matrix data, int layer) {
		Matrix next = weights[layer].parralel_mult(data);
		Vector v = biases[layer];
		for (int i = 0; i < v.length; i++) {
			for (int j = 0; j < next.width; j++) {
				next.v[i][j] += v.v[i];
			}
		}
		return next;
	}

	public Vector forward_layer(Vector data, int layer) {
		Vector next = weights[layer].dot(data);
		Vector v = biases[layer];
		for (int i = 0; i < v.length; i++) {
			next.v[i] += v.v[i];
		}
		return next;
	}

	public Matrix forward(Matrix data) {
		Matrix next = null;
		for (int k = 0; k < weights.length; k++) {
			if (k == 0)
				next = forward_layer(data, k);
			else
				next = forward_layer(next, k);

			if (k < weights.length - 1) {
				activation_forward(next);
			} else {
				Activations.softmax(next, 1);
			}
		}
		return next;
	}

	// Compte le nombre de points classifiés correctement
	public int correct_count(Matrix data, int[] refs) {
		int counter = 0;
		Matrix next = forward(data);
		for (int i = 0; i < data.width; i++) {
			Vector r = next.get_column(i);
			if (argmax(r) == refs[i]) {
				counter++;
			}
		}
		return counter;
	}

	// Calcule la correction à appliquer au poids en fonction d'un point
	public Gradient w_grad(Vector datapoint, int true_ref) {
		Gradient g = new Gradient();
		g.w = new Matrix[weights.length];
		g.b = new Vector[weights.length];

		Vector[] activations = new Vector[dims.length];
		activations[0] = datapoint;
		int c;
		for (c = 0; c < weights.length - 1; c++) {
			activations[c + 1] = activation_forward(forward_layer(activations[c], c));
		}
		activations[c + 1] = Activations.softmax(forward_layer(activations[c], c));
		c++;
		Vector dout = activations[c];
		dout.v[true_ref] -= 1;
		for (int k = weights.length - 1; k >= 0; k--) {
			if (k < weights.length - 1) {
				for (int i = 0; i < dout.length; i++) {
					dout.v[i] *= activation_backward(activations[k + 1].v[i]);
				}
			}
			g.w[k] = Vector.outer(dout, activations[k]).addInPlace(weights[k].scale(reg));
			g.b[k] = new Vector(dout);
			dout = weights[k].T().dot(dout);
		}
		return g;
	}

	// Calcule la correction à appliquer au poids en fonction d'un point
	public Gradient w_grad_vectorized(Matrix datapoints, int[] true_refs) {
		Gradient g = new Gradient();
		g.w = new Matrix[weights.length];
		g.b = new Vector[biases.length];

		Matrix[] activations = new Matrix[dims.length];
		activations[0] = datapoints;
		int c;
		for (c = 0; c < weights.length - 1; c++) {
			activations[c + 1] = activation_forward(forward_layer(activations[c], c));
		}
		activations[c + 1] = Activations.softmax(forward_layer(activations[c], c), 1);
		c++;

		Matrix dout = activations[c];
		for (int i = 0; i < true_refs.length; i++) {
			if (argmax(dout.get_column(i)) == true_refs[i]) {
				last_correct_count++;
			}
			dout.v[true_refs[i]][i] -= 1;
		}

		for (int k = weights.length - 1; k >= 0; k--) {
			if (k < weights.length - 1) {
				for (int j = 0; j < dout.height; j++) {
					for (int i = 0; i < dout.width; i++) {
						dout.v[j][i] *= activation_backward(activations[k + 1].v[j][i]);
					}
				}
			}
			g.w[k] = dout.parralel_mult(activations[k].T()).scaleInPlace(1.0 / datapoints.width)
					.addInPlace(weights[k].scale(reg));
			g.b[k] = dout.sum(1).scaleInPlace(1.0 / datapoints.width);

			if (k > 0) {
				dout = weights[k].T().parralel_mult(dout);
			}
		}
		return g;
	}

	public Matrix[] accelerations;
	public Vector[] b_accelerations;

	public final double gamma = 0.9;
	public int last_correct_count = 0;
	public int global_counter = 1;
	public final double epsilon = 1e-8;
	public final int mini_batch = 1;//512

	public void epoch_rmsprop() {
		last_correct_count = 0;
		ArrayList<Integer> ints = new ArrayList<Integer>();
		ArrayList<Vector> columns = new ArrayList<Vector>();
		for (int i = 0; i < data.width; i++) {
			ints.add(i);
			columns.add(data.get_column(i));
		}
		System.out.print("[");
		Collections.shuffle(ints);
		int a = (data.width / mini_batch) / 10;
		for (int i = 0; i < data.width / mini_batch; i++) {
			if (i % a == 0)
				System.out.print("=");
			//ImagePerceptron.visualizeClusterImage("sigvisu4/fig"+(global_counter++));
			Matrix batch = new Matrix(mini_batch, dims[0]);
			int[] refs_v = new int[mini_batch];
			for (int j = 0; j < mini_batch; j++) {
				int indice = ints.get(i * mini_batch + j);
				batch.set_column(j, columns.get(indice));
				refs_v[j] = refs[indice];
			}
			Gradient grads = w_grad_vectorized(batch, refs_v);
			if (accelerations == null) {
				accelerations = new Matrix[grads.w.length];
				b_accelerations = new Vector[grads.b.length];
				for (int k = 0; k < grads.w.length; k++) {
					accelerations[k] = new Matrix(grads.w[k].width, grads.w[k].height);
					b_accelerations[k] = new Vector(grads.b[k].length);
				}
			}
			for (int k = 0; k < grads.w.length; k++) {
				for (int l = 0; l < accelerations[k].height; l++) {
					for (int m = 0; m < accelerations[k].width; m++) {
						accelerations[k].v[l][m] = gamma * accelerations[k].v[l][m]
								+ (1 - gamma) * grads.w[k].v[l][m] * grads.w[k].v[l][m];
						grads.w[k].v[l][m] *= -learning_rate / (Math.sqrt(epsilon + accelerations[k].v[l][m]));
						weights[k].v[l][m] += grads.w[k].v[l][m];
					}
				}
				for (int l = 0; l < b_accelerations[k].length; l++) {
					b_accelerations[k].v[l] = gamma * b_accelerations[k].v[l]
							+ (1 - gamma) * grads.b[k].v[l] * grads.b[k].v[l];
					grads.b[k].v[l] *= -learning_rate / (Math.sqrt(epsilon + b_accelerations[k].v[l]));
					biases[k].v[l] += grads.b[k].v[l];
				}
			}
		}
		learning_rate *= learning_rate_decay;
		System.out.print("] ");
		//System.out.println(biases[0] + " " + biases[1]);
	}
	/*
	// Met à jour les poids en passant par toutes les données une fois
	public void epoch() {
		ArrayList<Integer> ints = new ArrayList<Integer>();
		for (int i = 0; i < data.width; i++) {
			ints.add(i);
		}
		Collections.shuffle(ints);
		for (int i = 0; i < data.width / mini_batch; i++) {
			Gradient grads = new Gradient();
			grads.w = new Matrix[weights.length];
			for (int k = 0; k < weights.length; k++) {
				grads.w[k] = new Matrix(weights[k].width, weights[k].height);
			}
			for (int j = 0; j < mini_batch; j++) {
				int indice = ints.get(i * mini_batch + j);
				Vector datapoint = data.getColumn(indice);
				int ref = refs[indice];
				Gradient grads_v = w_grad(datapoint, ref);
				for (int k = 0; k < weights.length; k++) {
					grads.w[k].addInPlace(grads_v.w[k]);
					grads.b[k].addInPlace(grads_v.b[k]);
				}
			}
			for (int k = 0; k < weights.length; k++) {
				grads.w[k].scaleInPlace(-learning_rate / mini_batch);
				weights[k].addInPlace(grads.w[k]);
				biases[k].addInPlace(grads.b[k].scaleInPlace(-learning_rate / mini_batch));
			}
		}
	}
	
	public void epoch_vectorized() {
		ArrayList<Integer> ints = new ArrayList<Integer>();
		ArrayList<Vector> columns = new ArrayList<Vector>();
		for (int i = 0; i < data.width; i++) {
			ints.add(i);
			columns.add(data.getColumn(i));
		}
		Collections.shuffle(ints);
		for (int i = 0; i < data.width / mini_batch; i++) {
			Matrix batch = new Matrix(mini_batch, dims[0]);
			int[] refs_v = new int[mini_batch];
			for (int j = 0; j < mini_batch; j++) {
				int indice = ints.get(i * mini_batch + j);
				batch.setColumn(j, columns.get(indice));
				refs_v[j] = refs[indice];
			}
			Gradient grads = w_grad_vectorized(batch, refs_v);
	
			if (i == 900) {
				int tocheck = 0;
				int[] titi = new int[1];
				int[] toto = new int[10];
				titi[0] = (int) (Math.random() * weights[tocheck].height);
	
				for (int k = 0; k < 10; k++) {
					do {
						toto[k] = (int) (Math.random() * weights[tocheck].width);
					} while (batch.v[toto[k]][0] < 0.1);
				}
				for (int a : titi) {
					for (int b : toto) {
						double epsilon = 1e-4;
						double old = weights[tocheck].v[a][b];
						weights[tocheck].v[a][b] = old + epsilon;
						Matrix next = null;
						for (int k = 0; k < weights.length; k++) {
							if (k == 0)
								next = weights[k].parralel_mult(batch);
							else
								next = weights[k].parralel_mult(next);
	
							if (k < weights.length - 1)
								activation_forward(next);
							else
								softmax(next, 1);
						}
	
						weights[tocheck].v[a][b] = old - epsilon;
						Matrix next2 = null;
						for (int k = 0; k < weights.length; k++) {
							if (k == 0)
								next2 = weights[k].parralel_mult(batch);
							else
								next2 = weights[k].parralel_mult(next2);
	
							if (k < weights.length - 1)
								activation_forward(next2);
							else
								softmax(next2, 1);
						}
						next2.printValues();
						weights[tocheck].v[a][b] = old;
	
						System.out.println("min & max of diff :" + next.add(next2.scale(-1)).min() + " "
								+ next.add(next2.scale(-1)).max());
	
						double loss1 = 0, loss2 = 0;
						for (int ref = 0; ref < refs_v.length; ref++) {
							loss1 += -Math.log(next.v[refs_v[ref]][ref]) / refs_v.length;
							loss2 += -Math.log(next2.v[refs_v[ref]][ref]) / refs_v.length;
						}
						System.out.println("losses : " + loss1 + "\t" + loss2 + "\t" + (loss1 - loss2));
	
						double analytic_grad = (loss1 - loss2) / (2 * epsilon);
	
						System.out.println(a + " " + b + " " + analytic_grad + " " + grads.w[tocheck].v[a][b] + " "
								+ Math.abs(analytic_grad - grads.w[tocheck].v[a][b])
										/ Math.abs(analytic_grad + grads.w[tocheck].v[a][b]));
						System.out.println("===============");
					}
				}
				System.exit(0);
			}
	
			for (int k = 0; k < grads.w.length; k++) {
				grads.w[k].scaleInPlace(-learning_rate);
				weights[k].addInPlace(grads.w[k]);
				grads.b[k].scaleInPlace(-learning_rate);
				biases[k].addInPlace(grads.b[k]);
			}
		}
	}
	*/
}