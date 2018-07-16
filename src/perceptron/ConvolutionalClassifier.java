package perceptron;

import java.util.ArrayList;
import java.util.Collections;

import layers.FeatureLayer;
import layers.FlatLayer;
import layers.Layer;
import layers.features.Flatten;
import layers.features.Unflatten;
import layers.losses.Loss;
import math.FeatureMatrix;
import math.Matrix;
import math.Vector;

@Deprecated
public class ConvolutionalClassifier extends ConvolutionalNetwork {

	public int last_correct_count = 0;
	
	public ConvolutionalClassifier(int mini_batch) {
		super(mini_batch);
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
			if (i % tenth == 0)
				System.out.print("=");
			
			Matrix next_m = new Matrix(1, data.height);
			FeatureMatrix next_fm = null;
			for (int j = 0; j < mini_batch; j++) {
				int indice = ints.get(i * mini_batch + j);
				Vector v = columns.get(indice);
				int ref = refs[indice];
				next_m.set_column(0, v);

				for(Layer l : layers) {
					if(l instanceof FlatLayer) {
						next_m = ((FlatLayer)l).forward(next_m, true);
						//System.out.println("applying " + l + " m is now "+next_m);
					} else if(l instanceof FeatureLayer) {
						next_fm = ((FeatureLayer)l).forward(next_fm, true);
						//System.out.println("applying "+l+" fm is now "+next_fm);
					} else if(l instanceof Layer) {
						if(l instanceof Flatten) {
							next_m = ((Flatten)l).forward(next_fm, true);
							//System.out.println("flattening to "+next_m);
						}
						if(l instanceof Unflatten) {
							next_fm = ((Unflatten)l).forward(next_m, true);
							//System.out.println("unflattening to "+next_fm);
						}
					} 
				}
				//long time = System.currentTimeMillis();
				for(double d : next_m.argmax(Matrix.AXIS_HEIGHT).add(new Vector(refs).scale(-1)).v) {
					last_correct_count += Math.abs(d) < 1e-8 ? 1 : 0;
				}
				
				get_loss_layer().feed_ref(Loss.from_int_refs(new int[] {ref}, next_m.height));
				for(int l_ind = layers.size()-1 ; l_ind >= 0 ; l_ind--) {
					Layer l = layers.get(l_ind);
					if(l instanceof FlatLayer) {
						next_m = ((FlatLayer)l).backward(next_m, true);
						//System.out.println("applying " + l + " m is now "+next_m);
					} else if(l instanceof FeatureLayer) {
						next_fm = ((FeatureLayer)l).backward(next_fm);
					} else if(l instanceof Layer) {
						if(l instanceof Flatten) {
							next_fm = ((Flatten)l).backward(next_m);
							//System.out.println("unflattening to "+next_fm);
						}
						if(l instanceof Unflatten) {
							next_m = ((Unflatten)l).backward(next_fm);
							//System.out.println("flattening to "+next_m);
						}
					} 
				}
				//System.out.println(System.currentTimeMillis()-time);
				last_average_loss += get_loss_layer().loss;
			}
			last_average_loss /= mini_batch;
			//System.out.println(last_correct_count);
			/*
			for(Layer l : layers) {
				if(l instanceof FlatLayer) {
					((FlatLayer)l).apply_gradient();
				} else if(l instanceof FeatureLayer) {
					((FeatureLayer)l).apply_gradient();
				}
			}
			*/
			
		}
		last_average_loss /= data.width / mini_batch;
		
		/*
		for(Layer l : layers) {
			if(l instanceof BatchnormLayer) {
				((BatchnormLayer)l).end_of_epoch();
			}
			if(l instanceof AffineLayer) {
				((AffineLayer)l).end_of_epoch();
			}
		}
		*/
		System.out.print("] ");
		//System.out.println(biases[0] + " " + biases[1]);
	}

	public double correct_count(Matrix data, int[] refs) {
		Matrix next = forward(data, false);
		int correct = 0;
		for(double d : next.argmax(Matrix.AXIS_HEIGHT).add(new Vector(refs).scale(-1)).v) {
			correct += Math.abs(d) < 1e-8 ? 1 : 0;
		}
		return correct;
	}
	
}
