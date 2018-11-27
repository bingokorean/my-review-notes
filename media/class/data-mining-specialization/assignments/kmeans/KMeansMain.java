import java.io.*;
import java.util.*;

public class KMeansMain {
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
        if (args.length != 2) { // 입력 데이터
            System.out.println("[usage] <data-file> <ground-truth-file>");
            // <data-file> -> self_test.data : [300 2] 300개 데이터들의 좌표
            // <ground-truth-file> -> self_test.ground : [300 1] 300개 데이터들의 라벨
            return;
        }
                
        String dataFilename = args[0];
        String groundtruthFilename = args[1];
        
        double[][] data = loadPoints(dataFilename);
        int[] groundtruth = loadClusters(groundtruthFilename);
        
        // 가정: centroid 개수를 2개로 미리 지정한다. (the number of centroids is 2 (fixed))
        // originally the number of centroids is determined by random
        double[][] centers = new double[2][data[0].length];	// (data[0].length = 2) -> 여기서 2는 좌표가 2개란 뜻.
        for (int i = 0; i < 2; ++ i) { // 여기서 2는 centroid가 2개
            for (int j = 0; j < data[i].length; ++ j) { // j 모두 2개이다. 좌표가 2개 이므로..
                centers[i][j] = data[i][j]; // center 초기화를 위해 간단하게 첫 번째, 두 번째 데이터의 좌표들을 할당. (원래는 랜덤이 더 적합함)
            }
        }
            
        // data = [300 x 2]
        // centers = [2 x 2] 
        int[] result = KMeans.kmeans(data, centers, 100, 1e-6);	// 100: 최대 iteration 횟수
        // result[K] = clusterID[K] , K = # of centroids
        
        System.out.println("# Purity = " + SupervisedEvaluation.purity(groundtruth, result));
        System.out.println("# NMI = " + SupervisedEvaluation.NMI(groundtruth, result));
    }
    
    public static void main(String[] args) {
        new KMeansMain().run(args);
    }
}
