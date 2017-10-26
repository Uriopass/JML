package image;

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
		double[] x  = new double[image.length*image[0].length];
		for (int i = 0; i < image.length; i++) {
			for (int j = 0; j < image[0].length; j++) {
				int k = image.length*i+j;
				x[k]=image[i][j]/255.0;
			}
		}
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
}
