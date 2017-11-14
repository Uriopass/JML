package image;

public class ImageConverter {
	public static double[] image2VecteurReel(int[][] image) {
		double[] x = new double[image.length * image[0].length];
		for (int i = 0; i < image.length; i++) {
			for (int j = 0; j < image[0].length; j++) {
				int k = image.length * i + j;
				x[k] = image[i][j] / 255.0;
			}
		}
		return x;
	}
}
