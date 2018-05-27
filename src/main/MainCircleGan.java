package main;

import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.util.List;
import java.util.Locale;

import javax.imageio.ImageIO;

import datareaders.MnistReader;
import gan.GenerativeAdversarialNetwork;
import image.ImageConverter;
import layers.Parameters;
import layers.flat.DenseLayer;
import layers.losses.SoftmaxCrossEntropy;
import math.Matrix;
import math.RandomGenerator;
import math.Vector;
import perceptron.MLPMetrics;
import perceptron.FlatSequential;

public class MainCircleGan {
	public static GenerativeAdversarialNetwork model;

	public final static int EPOCHMAX = 40000;

	// Number of training points
	public static final int N = 80;

	// Matrix containing the training daat
	public static Matrix train_data;
	
	// Seed used for reproducibility
	public static long seed;

	public static void load_data() {
		train_data = new Matrix(N, 2);

		// Creating data from random points on a circle
		for (int l = 0; l < N; l++) {
			double angle = RandomGenerator.uniform(0, 2*Math.PI);
			train_data.set_column(l, new Vector(new double[]{Math.cos(angle), Math.sin(angle)}));
		}

		System.out.println("# Train set built with " + N+ " points");
	}

	// Creating a matrix with a grid of points on [-2; 2] in x and y for the discriminator vizualisation
	public static Matrix batch = new Matrix(100*100, 2);;
	static {
		for(int i = 0 ; i < 100 ; i++) {
			for(int j = 0 ; j < 100 ; j++) {
				int id = j*100+i;
				batch.v[0][id] = 4f*i/100.0f-2f+1/50f;
				batch.v[1][id] = 4f*j/100.0f-2f+1/50f;
			}
		}
	}
	public static void visu_current(String name) {
		BufferedImage bf = new BufferedImage(1000, 1000, BufferedImage.TYPE_INT_ARGB);
		Matrix res = model.discriminator.forward(batch, false);

		// Visualizing discriminator
		Matrix points = train_data;
		for(int y = 0 ; y < 100 ; y++) {
			for(int x = 0 ; x < 100 ; x++) {
				float grey = (float) res.v[0][y*100+x];

				Color c = new Color(grey, grey, grey);
				
				//System.out.println(x + " " + y + "   " + c.getRed() +" " + c.getGreen() + " " + c.getBlue()+" "+train_data.v[y*32+x][0]);
				for(int yy = 0 ; yy < 10 ; yy++) {
					for(int xx=0; xx<10;xx++) {
						bf.setRGB(x*10+xx, y*10+yy, c.getRGB());
					}
				}
			}
		}
		// Visualizing generator
		Matrix res2 = model.test_generator();
		for(int i = 0 ; i < res2.width ; i++) {
			int x = (int)((res2.v[0][i]+2)*1000/4);
			int y = (int)((res2.v[1][i]+2)*1000/4);
			if(2 <= x && x < 998 && 2 <= y && y < 998) {
				for(int yy = -2 ; yy < 3 ; yy++) {
					for(int xx = -2 ; xx < 3 ; xx++) {
						bf.setRGB(x+xx, y+yy, Color.GREEN.getRGB());
					}
				}
			}
		}
		
		// Visualizing training data
		for(int i = 0 ; i < points.width ; i++) {
			int x = (int)((points.v[0][i]+2)*1000/4);
			int y = (int)((points.v[1][i]+2)*1000/4);
			if(2 <= x && x < 998 && 2 <= y && y < 998) {
				for(int yy = -2 ; yy < 3 ; yy++) {
					for(int xx = -2 ; xx < 3 ; xx++) {
						bf.setRGB(x+xx, y+yy, Color.blue.getRGB());
					}
				}
			}
		}
		
		try {
			ImageIO.write(bf, "png", new File(name+".png"));
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	public static void main(String[] args) {
		// On initialise le g�n�rateur al�atoire
		long time = System.currentTimeMillis();
		seed = 1510437982659L;
		RandomGenerator.init(seed);
		System.out.println("# Seed : " + seed);

		// Creating the networks
		model = new GenerativeAdversarialNetwork(2, 2);
		load_data();
		
		System.out.println("# Processors : " + Runtime.getRuntime().availableProcessors());

		// Useful to print numbers
		DecimalFormatSymbols otherSymbols = new DecimalFormatSymbols(Locale.getDefault());
		otherSymbols.setDecimalSeparator('.');
		otherSymbols.setGroupingSeparator(',');
		DecimalFormat df2 = new DecimalFormat("#0.00", otherSymbols);
		DecimalFormat df5 = new DecimalFormat("#0.00000", otherSymbols);

		System.out.println("# Initialization took " + (System.currentTimeMillis() - time) + " ms");

		int mini_batch = 80;
		
		for (int i = 1; i <= EPOCHMAX; i++) {
			long t = System.currentTimeMillis();
			Vector total_loss = new Vector(2);
			double l1=0, l2=0;
			
			for (int k = 0; k < train_data.width / mini_batch; k++) {

				l1 = model.train_discriminator(mini_batch, train_data, k*mini_batch, (k+1)*mini_batch);
				l2 = model.train_generator(mini_batch);
				
				total_loss.v[0] += l1;
				total_loss.v[1] += l2;
			}
			total_loss.scale(1. / (train_data.width / mini_batch));
			
			System.out.println(total_loss);

			// Time it took to execute one epoch
			double epoch_time = (System.currentTimeMillis() - t) / 1000.;

			t = System.currentTimeMillis(); 
			
			System.out.print(i + ((i >= 10) ? " " : "  "));
			System.out.print("epoch time " + df2.format(epoch_time) + "s ");

			// Time before doing it all
			System.out.println(" ETA " + df2.format((EPOCHMAX - i) * (epoch_time)) + "s");
			// Visualizing every 100 epochs
			if(i%100 == 0) { 
				visu_current("circlegan/"+i/100);
			}
		}
	}
}
