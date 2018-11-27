public class KernelKMeans
{
    public static double[][] kernel(double[][] data, double sigma) {
        // TODO transform the data points to kernel space
        // here we are going to implement RBF kernel, K(x_i, x_j) = e^{\frac{-|x_i - x_j|^2}{2 \sigma^2}}
    	double[][] matrix = new double[data.length][data.length];
    	
    	// Kernel의 핵심: transform to Kernel Space( N x N Matrix )
    	for (int i = 0; i < data.length; ++ i) {
    		for (int j = 0; j < data.length; ++ j) { 			
    			matrix[i][j] = Math.exp( -(squaredDistance(data,i,j)) / (2*Utils.sqr(sigma)) ); 
    		}
    	}
    	return matrix;
    }

    // To calculate distance, dot product or squared distance is used (it's necessary for using similarity function)
    // Similarity 함수 (e.g. dot product)
    private static double squaredDistance(double[][] data, int i, int j) {
	// TODO Auto-generated method stub	
	return (Utils.sqr(data[i][0] - data[j][0]) + Utils.sqr(data[i][1] - data[j][1]));
    }
}
