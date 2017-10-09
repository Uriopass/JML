package image;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

import javax.imageio.ImageIO;

import mnisttools.MnistReader;

public class ImageConverter {

	private static final float DEFAULT = 100;


	public static float[] image2VecteurBinaire(int[][] image, float seuil){
		float[] x  = new float[image.length*image[0].length];
		for (int i = 0; i < image.length; i++) {
			for (int j = 0; j < image[0].length; j++) {
				int k = image.length*i+j;
				if (image[i][j] > seuil){
					x[k] = 1;
				}
				else
					x[k] = 0;
			}
		}
		return x;
	}
	public static float[] image2VecteurReel(int[][] image){
		float[] x  = new float[image.length*image[0].length];
		for (int i = 0; i < image.length; i++) {
			for (int j = 0; j < image[0].length; j++) {
				int k = image.length*i+j;
				x[k]=image[i][j];
			}
		}
		return x;
	}
	
	public static double[] image2VecteurReel_withB(int[][] image){
		double[] x  = new double[image.length*image[0].length+1];
		for (int i = 0; i < image.length; i++) {
			for (int j = 0; j < image[0].length; j++) {
				int k = image.length*i+j+1;
				x[k]=image[i][j]/256.0;
			}
		}
		x[0] = 1;
		return x;
	}
	
	
	public static int[] float2IntVector(float []x){
		int[] y = new int[x.length];
		for (int i = 0; i < y.length; i++) {
			y[i] = (int) x[i];
		}
		return y;
	}
	
	public static int[] image2IntVector(int[][] image){
		int[] x  = new int[image.length*image[0].length];

		for (int i = 0; i < image.length; i++) {
			for (int j = 0; j < image[0].length; j++) {
				int k = image.length*i+j;
				x[k] = image[i][j];
			}
		}
		return x;
	}
	
	public static float[] image2VecteurBinaire(int[][] image){
		return image2VecteurBinaire(image, DEFAULT);
	}


	public static void main(String[] args) {
		String path="./";
		String labelDB=path+"train-labels-idx1-ubyte";
		String imageDB=path+"train-images-idx3-ubyte";
		MnistReader db = new MnistReader(labelDB, imageDB);
		int idx = 1; // une variable pour stocker l'index 
		// Attention la premiere valeur est 1. 
		int [][] image = db.getImage(idx);
		float[] x = image2VecteurBinaire(image);
		int l = 0;
		int sizeL = image[l].length;
		for (int i = 0; i < sizeL ; i++) {
			//System.out.print(x[i+l*sizeL]+", ");
			assert(x[i+l*sizeL] == image[l][i]);
		}
		System.exit(0);
		System.out.println("ici");
		//System.out.println(Arrays.toString(image[l]));
		int[] vimage = float2IntVector(x); //image2intVector(image);
		int numberOfColumns=28;//image.length;
		int numberOfRows = 28; //image[0].length;
		BufferedImage bimage = new BufferedImage(numberOfColumns, numberOfRows, BufferedImage.TYPE_INT_ARGB);
		bimage.setRGB(0, 0, numberOfColumns, numberOfRows, vimage, 0, numberOfColumns);
		File outputfile = new File(path+"imagesChiffresMannuscrits.png");
		System.out.println("s");
		try {
			System.out.println("l�");
			ImageIO.write(bimage, "png", outputfile);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			System.out.println("l�");
			e.printStackTrace();
		}
	}
}
