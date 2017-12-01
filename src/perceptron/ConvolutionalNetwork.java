package perceptron;

import java.util.ArrayList;

import layers.FeatureLayer;
import layers.FlatLayer;
import layers.Layer;
import layers.features.Flatten;
import layers.features.Unflatten;
import layers.losses.Loss;
import math.FeatureMatrix;
import math.Matrix;

public class ConvolutionalNetwork extends FeedForwardNetwork {
	ArrayList<Layer> layers;
	int mini_batch = 0;
	public ConvolutionalNetwork(int mini_batch) {
		this.mini_batch = mini_batch;
		layers = new ArrayList<>();
	}
	
	public void addLayer(Layer l) {
		layers.add(l);
	}
	
	@Override
	public Matrix forward(Matrix data, boolean train) {
		Matrix result = null;
		for(int i = 0 ; i < data.width ; i++) {
			Matrix d = new Matrix(1, data.height);
			d.set_column(0, data.get_column(i));
			Matrix next_m = d;
			FeatureMatrix next_fm = null;
			
			for(Layer l : layers) {
				if(l instanceof FlatLayer) {
					next_m = ((FlatLayer)l).forward(next_m, false);
				} else if(l instanceof FeatureLayer) {
					next_fm = ((FeatureLayer)l).forward(next_fm, false);
				} else if(l instanceof Layer) {
					if(l instanceof Flatten) {
						next_m = ((Flatten)l).forward(next_fm, false);
					}
					if(l instanceof Unflatten) {
						next_fm = ((Unflatten)l).forward(next_m, false);
					}
				} 
			}
			if(result == null) {
				result = new Matrix(data.width, next_m.height);
			}
			result.set_column(i, next_m.get_column(0));
		}
		return result;
	}

	@Override
	public Loss get_loss_layer() {
		return (Loss)layers.get(layers.size()-1);
	}
	
	@Override
	public void print_architecture() {
		for(Layer l : layers) {
			System.out.println("# - "+l);
		}
	}
}
