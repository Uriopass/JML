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
import layers.losses.Loss;
import layers.losses.SoftmaxCrossEntropy;
import math.Matrix;
import math.Vector;

public class MultiLayerPerceptron extends FeedForwardNetwork {

	public ArrayList<FlatLayer> layers;
	
	public void add(FlatLayer l) {
		layers.add(l);
	}
	
	public int global_counter = 1;
	public int last_correct_count = 0;

	public MultiLayerPerceptron(int batch_size) {
		layers = new ArrayList<FlatLayer>();
		this.mini_batch = batch_size;
	}
	
	public MultiLayerPerceptron(int[] dims) {
		init_layers(dims, true);
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
		last_average_loss = 0;
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
			if(tenth != 0) {
				if (i % tenth == 0)
					System.out.print("=");
			}
			
			//ImagePerceptron.visualizeClusterImage("sigvisu4/fig"+(global_counter++));
			//System.out.println("start");
			for (int j = 0; j < mini_batch; j++) {
				int indice = ints.get(i * mini_batch + j);
				batch.set_column(j, columns.get(indice));
				refs_v[j] = refs[indice];
			}
			
			batch = forward_train(batch);
			
			for(double d : batch.argmax(Matrix.AXIS_HEIGHT).add(new Vector(refs_v).scale(-1)).v) {
				if(Math.abs(d) < 1e-8) {
					last_correct_count += 1;
				}
			}
			
			Matrix dout = batch;
			Loss l = getLoss();
			l.feed_ref(Loss.from_int_refs(refs_v, dout.height));
			
			backward_train(dout);
			
			last_average_loss += l.loss;
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
	
	public Matrix confusion_matrix(Matrix data, int[] refs) {
		Matrix end = forward(data);
		Matrix confusion_matrix = new Matrix(end.height, end.height);
		
		for(int i = 0 ; i < refs.length ; i++) {
			int correct = refs[i];
			int predicted = end.get_column(i).argmax();
			confusion_matrix.v[predicted][correct] += 1;
		}
		
		return confusion_matrix;
	}

	public Matrix k_wrongest_data(Matrix data, int[] refs, int k) {
		Matrix end = forward(data);
		
		Matrix wrongest = new Matrix(k, data.height);
		
		Vector wrongV = new Vector(data.width);
		for(int i = 0 ; i < data.width ; i++) {
			wrongV.v[i] = 1 - end.get_column(i).v[refs[i]];
		}
		
		for(int i = 0 ; i < k ; i++) {
			int wrongest_i = wrongV.argmax();
			System.out.println(wrongV.v[wrongest_i]);
			wrongest.set_column(i, data.get_column(wrongest_i));
			wrongV.v[wrongest_i] = 0;
		}
		
		return wrongest;
	}
	
	public Matrix average_data_by_class(Matrix data) {
		Matrix end = forward(data);
		Matrix average = new Matrix(end.height, end.height);
		
		for(int i = 0 ; i < end.width ; i++) {
			Vector v = end.get_column(i);
			int argmax = v.argmax();
			average.set_column(argmax, average.get_column(argmax).add(v));
		}
		average.scale(end.width);
		
		return average;
	}
	
	// Compte le nombre de points classifiés correctement
	public int correct_count(Matrix data, int[] refs) {
		Matrix next = forward(data);
		int zeros = 0;
		for(double d : next.argmax(Matrix.AXIS_HEIGHT).add(new Vector(refs).scale(-1)).v) {
			zeros += Math.abs(d) < 1e-8 ? 1 : 0;
		}
		return zeros;
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
	public Loss getLoss() {
		return (Loss)layers.get(layers.size()-1);
	}
	@Override
	public void print_architecture() {
		for(Layer l : layers) {
			System.out.println("# - "+l);
		}
	}
}
