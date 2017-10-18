package perceptron;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;

import javax.imageio.ImageIO;

import layers.ImageEntropyLoss;
import layers.Layer;
import layers.flatlayers.AffineLayer;
import layers.flatlayers.BatchnormLayer;
import math.Matrix;
import math.Vector;

public class AutoEncoder extends FeedForwardNetwork {

	@Override
	public Matrix forward(Matrix data) {
		Matrix next = new Matrix(data);
		for(Layer l : layers) {
			next = l.forward(next, false);
		}
		return next;
	}
	
	public void writeDiff(Matrix data, String name, int num) {
		Matrix result = forward(data);
		int dimension = 28;
		int scale = 1;
		int height = num;
		BufferedImage bf = new BufferedImage(dimension*scale*2, dimension*scale*height, BufferedImage.TYPE_INT_ARGB);
		for(int i = 0 ; i < height ; i++) {
			for(int j = 0 ; j < dimension*scale ; j++) {
				for(int k = 0 ; k < dimension*scale*2 ; k++) {
					int indice = dimension*(j/scale)+(k%(dimension*scale))/scale;
					int r, g, b;
					if(k >= dimension*scale) {
						r = (int)(255*data.v[indice][i]);
						g = (int)(255*data.v[indice][i]);
						b = (int)(255*data.v[indice][i]);
					} else {
						r = (int)(255*result.v[indice][i]);
						g = (int)(255*result.v[indice][i]);
						b = (int)(255*result.v[indice][i]);
					}
					
					int rgb = (0xFF << 24) + (g << 8) + (r << 16)+b;
					bf.setRGB(k, i*dimension*scale + j, rgb);
				}
			}
		}
		try {
			ImageIO.write(bf, "png", new File(name+".png"));
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	public void epoch(Matrix data) {
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
			if (i % tenth == 0)
				System.out.print("=");
			
			//ImagePerceptron.visualizeClusterImage("sigvisu4/fig"+(global_counter++));
			//System.out.println("start");
			for (int j = 0; j < mini_batch; j++) {
				int indice = ints.get(i * mini_batch + j);
				batch.set_column(j, columns.get(indice));
			}
			
			Matrix batch_init = new Matrix(batch);
			
			for(int j = 0 ; j < layers.size() ; j++) {
				batch = layers.get(j).forward(batch, true);
			}
			
			Matrix dout = batch;
			((ImageEntropyLoss) layers.get(layers.size()-1)).feed_ref(batch_init);
			
			for(int j = layers.size()-1 ; j >= 0 ; j--) {
				dout = layers.get(j).backward(dout);
				layers.get(j).apply_gradient();
			}
			last_average_loss += ((ImageEntropyLoss) layers.get(layers.size()-1)).loss;
		}
		last_average_loss /= data.width / mini_batch;
		
		for(Layer l : layers) {
			if(l instanceof BatchnormLayer) {
				((BatchnormLayer)l).end_of_epoch();
			}
			if(l instanceof AffineLayer) {
				((AffineLayer)l).end_of_epoch();
			}
		}
		System.out.print("] ");
	}

}
