package perceptron;

import java.io.File;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Scanner;

import layers.FlatLayer;
import layers.Layer;
import layers.Parameters;
import layers.activations.TanhActivation;
import layers.flat.AffineLayer;
import layers.flat.BatchnormLayer;
import layers.flat.SoftmaxCrossEntropy;
import math.Matrix;
import math.Vector;

public class MultiLayerPerceptron extends FeedForwardNetwork {

	public ArrayList<FlatLayer> layers;
	
	@Override
	public void add(Layer l) {
		layers.add((FlatLayer)l);
	}
	
	public int global_counter = 1;
	public int last_correct_count = 0;

	
	public MultiLayerPerceptron(int...dims) {
		init_layers(dims, true);
	}
	
	public MultiLayerPerceptron() {
		layers = new ArrayList<FlatLayer>();
	}
	
	/**
	 * @param name File name
	 */
	public MultiLayerPerceptron(String name) {
		try {
			File f = new File(name + ".jml");
			if (!f.exists()) {
				throw new RuntimeException("No file named" + name);
			}
			System.out.println("# Loading weights in "+name+".jml");

			Scanner sc = new Scanner(f);
			int dimslength = sc.nextInt();
			
			int[] dims = new int[dimslength];
			for (int i = 0; i < dims.length; i++) {
				int dimsi = sc.nextInt();
				dims[i] = dimsi;
			}
			
			init_layers(dims, false);
			ArrayList<AffineLayer> afflayers = new ArrayList<AffineLayer>();
			for(FlatLayer l : layers) {
				if(l instanceof AffineLayer) {
					afflayers.add((AffineLayer) l);
				}
			}
			
			for (int i = 0; i < dims.length - 1; i++) {
				for (int k = 0; k < dims[i+1]; k++) {
					for (int j = 0; j < dims[i]; j++) {
						//System.out.println(i+" "+k+" "+j+" "+sc.hasNextDouble());
						afflayers.get(i).weight.v[k][j] = Double.parseDouble(sc.next());
					}
				}
				Vector b = afflayers.get(i).bias;
				for (int k = 0; k < b.length; k++) {
					b.v[k] = Double.parseDouble(sc.next());
				}
			}
			sc.close();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	
	public void init_layers(int[] dims, boolean init) {
		double learning_rate = 0.001;
		double learning_rate_decay = 1;//0.794328235; // x^10 = 0.1
		double reg = 0.00003;
		
		layers = new ArrayList<FlatLayer>();
		Parameters p = new Parameters("lr="+learning_rate, "lrdecay="+learning_rate_decay, "reg="+reg);
		for(int i = 0 ; i < dims.length-1 ; i++) {
			if(i == 0)
				p.set("dout", "false");
			else
				p.set("dout", "true");
			
			layers.add(new AffineLayer(dims[i], dims[i+1], init, p));
			layers.add(new BatchnormLayer(dims[i+1], p));
			if(i < dims.length-2)
				layers.add(new TanhActivation());
		}
		layers.add(new SoftmaxCrossEntropy());
		
		System.out.println("# Model created with following architecture : ");
		print_architecture();
	}
	
	@Override
	public Matrix forward(Matrix data) {
		Matrix next = new Matrix(data);
		for(FlatLayer l : layers) {
			next = l.forward(next, false);
		}
		return next;
	}
	
	public Matrix forward_train(Matrix data) {
		Matrix next = new Matrix(data);
		for(FlatLayer l : layers) {
			next = l.forward(next, true);
		}
		return next;
	}
	
	public void backward_train(Matrix dout) {
		for(int j = layers.size()-1 ; j >= 0 ; j--) {
			dout = layers.get(j).backward(dout);
			layers.get(j).apply_gradient();
		}
	}
	
	public void epoch(Matrix data, int[] refs) {
		last_correct_count = 0;
		ArrayList<Integer> ints = new ArrayList<Integer>();
		ArrayList<Vector> columns = new ArrayList<Vector>();
		
		for (int i = 0; i < data.width; i++) {
			ints.add(i);
			columns.add(data.get_column(i));
		}
		Collections.shuffle(ints);
		
		System.out.print("[");
		int tenth = (data.width / mini_batch) / 10;
		
		for (int i = 0; i < data.width / mini_batch; i++) {
			Matrix batch = new Matrix(mini_batch, data.height);
			int[] refs_v = new int[mini_batch];
			
			if (i % tenth == 0)
				System.out.print("=");
			
			//ImagePerceptron.visualizeClusterImage("sigvisu4/fig"+(global_counter++));
			//System.out.println("start");
			for (int j = 0; j < mini_batch; j++) {
				int indice = ints.get(i * mini_batch + j);
				batch.set_column(j, columns.get(indice));
				refs_v[j] = refs[indice];
			}
			
			batch = forward_train(batch);
			
			Matrix dout = batch;
			((SoftmaxCrossEntropy) layers.get(layers.size()-1)).feedrefs(refs_v);
			
			backward_train(dout);
			last_correct_count += ((SoftmaxCrossEntropy) layers.get(layers.size()-1)).correct;
			last_average_loss += ((SoftmaxCrossEntropy) layers.get(layers.size()-1)).loss;
		}
		last_average_loss /= data.width / mini_batch;
		
		for(FlatLayer l : layers) {
			if(l instanceof BatchnormLayer) {
				((BatchnormLayer)l).end_of_epoch();
			}
			if(l instanceof AffineLayer) {
				((AffineLayer)l).end_of_epoch();
			}
		}
		System.out.print("] ");
		//System.out.println(biases[0] + " " + biases[1]);
	}

	// Compte le nombre de points classifiés correctement
	public int correct_count(Matrix data, int[] refs) {
		Matrix next = forward(data);
		SoftmaxCrossEntropy sf = ((SoftmaxCrossEntropy)(layers.get(layers.size()-1)));
		sf.feedrefs(refs);
		sf.backward(next);
		return sf.correct;
	}
	
	
	public void write_weights(String name) {
		try {
			File f = new File(name + ".jml");
			if (!f.exists()) {
				f.createNewFile();
			}
			PrintWriter pw = new PrintWriter(f);
			ArrayList<AffineLayer> afflayers = new ArrayList<AffineLayer>();
			for(FlatLayer l : layers) {
				if(l instanceof AffineLayer) {
					afflayers.add((AffineLayer) l);
				}
			}
			
			pw.write("" + (afflayers.size()+1));
			pw.write('\n');
			pw.write("" + afflayers.get(0).fan_in);
			for (int i = 0; i < afflayers.size(); i++) {
				pw.write(" "+afflayers.get(i).fan_out);
			}
			pw.write('\n');

			for (int i = 0; i < afflayers.size(); i++) {
				Matrix m = afflayers.get(i).weight;
				for (int k = 0; k < m.height; k++) {
					for (int j = 0; j < m.width; j++) {
						pw.write(m.v[k][j] + " ");
					}
					pw.write('\n');
				}
				Vector b = afflayers.get(i).bias;
				for (int k = 0; k < b.length; k++) {
					pw.write(b.v[k] + " ");
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
	
	@Override
	public void print_architecture() {
		for(Layer l : layers) {
			System.out.println("# - "+l);
		}
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
