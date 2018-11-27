public class SupervisedEvaluation
{
	
    public static double purity(int[] groundtruthAssignment, int[] algorithmAssignment) {
        // TODO compute the purity
    	double max_temp=0,length_temp=0;

    	double[][] c = new double[2][2];
    	double[] max = new double[2];
    	
	// External Measure의 기본인 (C,T)Matrix를 구현해야 한다. 그리고 해당 cluster에서 max를 구한다.
	// Clustering 갯수를 인자로 못받으므로, 여기서는 그냥 임위로 설정해준다. 
	// # of centroids or clusterings = 2
	for ( int k=0; k < 2; ++k ){	// k = #clustering(2)	
		for (int i = 0; i < groundtruthAssignment.length; ++ i ) {	// #alldata(300)
			if( algorithmAssignment[i] == k ) {	// k번째 clustering의 기준을 잡는다. Matrix는 clustering을 기준으로 만든다.					
				if(groundtruthAssignment[i] == 0)		
					c[k][0] += 1; // 해당 cluster에 ground true가 0인게 몇개 있느냐
				else if(groundtruthAssignment[i] == 1)	
					c[k][1] += 1; // 해당 cluster에 ground true가 1인게 몇개 있느냐				
				//else if... ground true 갯수만큼 게속 추가해준다. (여기서는 2개)
			} 		
		}
	}
	    	
	// Max값 선택	
	for( int i=0; i < 2; i++ ) {
	    if(c[i][0] > c[i][1] )	
		    max[i] = c[i][0];
	    else	
		    max[i] = c[i][1];
	    		
	}
	   
//             (C,T) Matrix (K-Means)
//	    	13   89
//	    	87   111
// *max = 89와 111인데,,  서로 다른 2개의 clustering이 공통으로 하나의 ground true를 max로 잡고 있다. 이럴수도 있다... evaluation하는 과정이니까..
// 즉, 현재 상황은 clustering이 안좋게 된 상황인데(이상적인 것은 cluster 개당 각각 서로다른 ground true를 max값으로 가져야한다.), purity로는 그대로 꽤 높은 점수 0.66를 얻었다. 그렇게 효용성 있지는 않다. (NMI와 비교할것)
 
	max_temp=max[0]+max[1];	// c0과 c1의 최대 ground true값
	length_temp=groundtruthAssignment.length;

        return max_temp/length_temp;
    }
  
    public static double log2(double num)	// java에서는 log2를 제공해주는 함수가 없다. 따로 구현필요
    {
    	return (Math.log(num)/Math.log(2));
    }    
    

    public static double NMI(int[] groundtruthAssignment, int[] algorithmAssignment) {
    	double h_t=0,h_c=0,i_ct=0,nmi_ct=0;
    	double[][] c = new double[2][2]; double[] max = new double[2];
    	double[] m = new double[2]; double[] n = new double[2];
    	double total = 0, epsilon=0.00000000001;
    		
	for ( int k=0; k < 2; ++k ){	
		for (int i = 0; i < groundtruthAssignment.length; ++ i ) {	
			if( algorithmAssignment[i] == k ) {					
				if(groundtruthAssignment[i] == 0)		c[k][0] += 1;	
				else if(groundtruthAssignment[i] == 1)	c[k][1] += 1;					
			} 		
		}
	}
	    	
	for( int i=0; i < 2; i++ ) {
	    if(c[i][0] > c[i][1] )	max[i] = c[i][0];
	    else	max[i] = c[i][1];
	    		
	}
    	
	// -- 여기까지 purity와 동일 .. 똑같은 Exteral Measure이다 보니, 기반은 비슷함..  (C,T) Matrix에서 시작한다 -- //
	    
        System.out.println("  ----- (C,T) Matrix ----- ");
        System.out.println("      "+c[0][0]+"     "+c[0][1]);
        System.out.println("      "+c[1][0]+"     "+c[1][1]);
        System.out.println("  ----- ------------ ----- ");

	    
//             (C,T) Matrix
//    		13      89	    (102) n[0]
//    		87     111	    (198) n[1]
//	    	(100) (200)	    (300)
//		m[0]	m[1]
	    
	m[0] = c[0][0]+c[1][0];		n[0] = c[0][0]+c[0][1];
	m[1] = c[0][1]+c[1][1];		n[1] = c[1][0]+c[1][1];
	total = m[0] + m[1];
	       
	// Compute the normalized mutual information.	(chaper17참고)
	h_t = -( (m[0]/total)*log2((m[0])/total) + ((m[1])/total)*log2((m[1])/total) ); 
	h_c = -( (n[0]/total)*log2(n[0]/total) + (n[1]/total)*log2(n[1]/total) );
	    
	i_ct =  ((c[0][0]/total)* log2( c[0][0]*total/ (m[0]*n[0]) + epsilon ) ) + 
	   	((c[0][1]/total)* log2( c[0][1]*total/ (m[1]*n[0]) + epsilon ) ) +
	    	((c[1][0]/total)* log2( c[1][0]*total/ (m[0]*n[1]) + epsilon ) ) +
	    	((c[1][1]/total)* log2( c[1][1]*total/ (m[1]*n[1]) + epsilon ) );
	    
        System.out.println("  -h_t "+h_t);
        System.out.println("  -h_c "+h_c);
        System.out.println("  -i_ct "+i_ct);
	    
	nmi_ct = i_ct/(Math.sqrt(h_t*h_c));
        return nmi_ct;
    }
}
