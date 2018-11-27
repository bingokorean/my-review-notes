public class KMeans {
	
    // SSE (Sum of Squared Error) 함수 정의
    public static double computeSSE(double[][] data, double[][] centers, int[] clusterID) { 
        double sse = 0;
        int n = data.length;
        for (int i = 0; i < n; ++ i) {
            int c = clusterID[i];
            sse += Utils.squaredDistance(data[i], centers[c]); // SSE(C) function formulation (K-Means & Kernel K-Means)
        }
        return sse;
    }
    
    
    public static int[] updateClusterID(double[][] data, double[][] centers) {
    	
    	/// Shape of centers (i.e. centers.length)
    	// K-Means - [ 2 x 2 ] : x, y axis
    	// Kernel K-Means - [ 2 x 300 ] : high dimensional features
    	
    	int[] clusterID = new int[data.length];
	double[] temp_error = new double[centers.length];
	double min_value=0, min_idx=0, temp_sum=0;
        // TODO assign the closet center to each data point
        // you can use the function Utils.squaredDistance		
	for (int i = 0; i < data.length; ++ i) {	// # of all data 
		for (int k = 0; k < centers.length; ++ k) {	// # of centroids
			temp_error[k] = Utils.squaredDistance( data[i], centers[k] ); // calculate distance error
		}
			
		// 매 데이터마다 모든 centroid들과의 거리를 비교하고, 
		// 가장 가까운 cenroid ID (=min_idx)를 해당 데이터(=clusterID[i])에게 부여함.
			
		// minimum temp_error among centroids
		for (int j = 1; j < centers.length; ++ j) {	// array boundary
			if(temp_error[j-1] < temp_error[j])	{
				min_value = temp_error[j-1];
				min_idx = j-1;
			}
			else {
				min_value = temp_error[j];
				min_idx = j;
			}
		}			
		clusterID[i] = (int) min_idx; // the point is assigned by the closest centorid id (MATLAB -> clusterID = idx)
			
    	}
        return clusterID;
    }
    
    public static double[][] updateCenters(double[][] data, int[] clusterID, int K) {
	    
        // TODO recompute the centers based on current clustering assignment
        // If a cluster doesn't have any data points, in this homework, leave it to ALL 0s.
        double[][] centers = new double[K][data[0].length];
        double[][] sum = new double[K][data[0].length];
        int[] count = new int[K]; // K는 centroid 개수이다.
        
	for (int i = 0; i < data.length; ++ i) {
		for (int k = 0; k < K; ++ k) {
			if( clusterID[i] == k )	{
				for (int n = 0; n < data[0].length; ++ n)	sum[k][n] = sum[k][n] + data[i][n];
				count[k] = 1 + count[k];					
			}
		}			
        }
		
	for (int k = 0; k < K; ++ k) {
		 for (int n = 0; n < data[0].length; ++ n) {				 
			 if(count[k] != 0)	centers[k][n] = sum[k][n]/count[k]; // for each centroid, calculate average coordinates
		 }
	}		
        return centers; // return new centers!
    }
    
    /** run kmeans a single time, with specific initialization and number of iterations
     *  double[][] data are the points need to be clustered
     *  double[][] centers are the initializations
     *  int maxIter is the max number of itertions for kmeans
     *  double tol is the tolerance for stop criterion
     *  return clusterID
    **/
    public static int[] kmeans(double[][] data, double[][] centers, int maxIter, double tol) {
        int n = data.length; // the number of data points
        if (n == 0) {
            return new int[0];
        }
        int k = centers.length;
		
        int[] clusterID = new int[n];
        if (k >= n) {
            for (int i = 0; i < n; ++ i) {
                clusterID[i] = i;
            }
			System.out.println("DEBUG : keams if(k>=n)");
            return clusterID;
        }

        double lastDistance = 1e100; // set to infinity
        for (int iter = 0; iter < maxIter; ++ iter) {
            clusterID = updateClusterID(data, centers);
            centers = updateCenters(data, clusterID, k);
            double distance = computeSSE(data, centers, clusterID);
            
	    // 최적화 방법.
            if ((lastDistance - distance) < tol || (lastDistance - distance) / lastDistance < tol) { 
            	System.out.println("# iterations = " + iter);
                System.out.println("SSE = " + distance);
                return clusterID;
            }
            lastDistance = distance;
        }
        System.out.println("# iterations = " + maxIter);
        System.out.println("SSE = " + lastDistance);
        return clusterID;
    }
}
