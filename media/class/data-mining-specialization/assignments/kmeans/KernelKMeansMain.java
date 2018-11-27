import java.io.*;
import java.util.*;

public class KernelKMeansMain {
    double[][] loadPoints(String filename) {
        try {
            Scanner sc = new Scanner(new File(filename));
            int n = sc.nextInt();
            int d = sc.nextInt();
            double[][] data = new double[n][d];
            for (int i = 0; i < n; ++ i) {
                for (int j = 0; j < d; ++ j) {
                    data[i][j] = sc.nextDouble();
                }
            }
            return data;
        } catch (Exception e) {
            e.printStackTrace();
            System.err.println("error while reading data points file");
        }
        return null;
    }
    
    int[] loadClusters(String filename) {
        try {
            Scanner sc = new Scanner(new File(filename));
            int n = sc.nextInt();
            int[] clusterID = new int[n];
            for (int i = 0; i < n; ++ i) {
                clusterID[i] = sc.nextInt();
            }
            return clusterID;
        } catch (Exception e) {
            e.printStackTrace();
            System.err.println("error while reading cluster file");
        }
        return null;
    }
    
    void run(String[] args) {
        if (args.length != 2) {
            System.out.println("[usage] <data-file> <ground-truth-file>");
            return;
        }
        
        String dataFilename = args[0];
        String groundtruthFilename = args[1];
        
        double[][] data = loadPoints(dataFilename);
        int[] groundtruth = loadClusters(groundtruthFilename);
        
        final double SIGMA = 4;
        
        /// Kernel K-means 핵심.
        // data is transformed from original space to kernel space
        // data space moving [ 300 x 2 ] -> [ 300 x 300 ]
        data = KernelKMeans.kernel(data, SIGMA);
       
        
        double[][] centers = new double[2][data[0].length]; // data[0].length = 300 (이제 x,y 좌표가 아니다)
        for (int i = 0; i < 2; ++ i) { // # centroid 개수는 여전히 2개로 고정시킨다.
            for (int j = 0; j < data[i].length; ++ j) {
                centers[i][j] = data[i][j]; // center 초기화를 위해서 간단하게 첫 번째, 두 번째 데이터 좌표들을 그대로 사용.
            }
        }
        
        int[] result = KMeans.kmeans(data, centers, 100, 1e-6);
        
        System.out.println("# Purity = " + SupervisedEvaluation.purity(groundtruth, result));
        System.out.println("# NMI = " + SupervisedEvaluation.NMI(groundtruth, result));
    }
    
    public static void main(String[] args) {
        (new KernelKMeansMain()).run(args);
    }
}
