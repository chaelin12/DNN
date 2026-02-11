package layers;

import java.util.*;

public class Operations {


    public static void calculateWeightedSum(double[] inputs, double[][] weights, double[] output) {
        int m = inputs.length;//입력 벡터의 길이
        int w = weights.length;//행렬의 행 개수(입력 쪽 가중치 개수)
        int n = weights[0].length;//신호를 받는 뉴런 개수 = output 개수

        Arrays.fill(output, 0.0);//output을 0으로 채움

        if ( m != w )//입력 행과 출력 열 개수가 같아야 행렬 곱 가능
        {
            System.err.println("행렬 곱 연산 불가");
            return;
        }

        for (int i = 0; i < m; i++) {//입력 노드 하나하나
            if (inputs[i] == 0) continue; // 0인 입력은 계산 건너뛰기( 어차피 0이니까 )
            double inputVal = inputs[i];
            for (int j = 0; j < n; j++) {//output 개수 만큼
                output[j] += inputVal * weights[i][j];//W*x, 입력 노드 * 노드에서 뻗어나가는 가중치를 모두 곱한 값이 출력 모든 output 노드에 누적 저장됨
            }//행부터 계산 후 열로 넘어가므로 속도 빨라짐
        }
    }

    //델타 계산: dL_dz = dL_da * da_dz
    public static void calculateDelta(double[] dL_da, double[] da_dz, double[] dL_dz) {
        if (dL_da.length != da_dz.length || dL_da.length != dL_dz.length) {
            System.err.println("행렬 곱 연산 불가");
            return;
        }
        for (int i = 0; i < dL_dz.length; i++) {
            dL_dz[i] = dL_da[i] * da_dz[i];
        }
    }

    //가중치 기울기 계산: dL_dw += dL_dz * Input
    public static void calculateWeightGradient(double[] input, double[] dL_dz, double[][] dL_dw) {
        int m = input.length;//입력 개수
        int n = dL_dz.length;//오차가 얼마나 난지 

        // dL_dw[j][i] += dL_dz[j] * input[i]
        for ( int i = 0; i < m; i++ )
            for ( int j = 0; j < n; j++ )
                dL_dw[ i ][ j ] += input[ i ] * dL_dz[ j ]; // i에서 j번 노드로 연결된 edge마다 가중치 기울기 저장
    
    }//미니배치 사용, batch 사이즈 만큼 이미지 가중치 기울기 누적 저장 후 평균을 내서 업데이트


    //이전 층 오차 전파: dL_dx += Weights^T * dL_dz
    public static void calculatePrevLayerError(double[] dL_dz, double[][] weights, double[] dL_dx) {
        int n = dL_dz.length;//오차 책임(오차의 개수)
        int m = weights.length;//연결된 가중치의 개수(앞 층의 뉴런 개수)
        int nn = weights[0].length;//내부 가중치의 개수 = 다음 층의 크기(노드의 개수)
        
        if ( n != nn )//오차랑 연결지을 가중치의 개수가 맞지 않으면
        {
            System.err.println("행렬 곱 연산 불가");
            return;
        }

        Arrays.fill( dL_dx, 0.0 );
        for ( int i = 0; i < m; i++ )//i번째 노드는 다음층의 모든 노드와 연결, 영향
        //이전 층의 노드 인덱스
            for ( int j = 0; j < n; j++ ) 
                //현재 층의 노드 인덱스
                dL_dx[ i ] += dL_dz[ j ] * weights[ i ][ j ];//전달받은 노드의 오차 * 가중치 -> 앞층의 i번째 노드의 dL_da를 구함
                //행렬 모양으로 보면 이 과정이 가중치 행렬을 뒤집어서 곱하는 것과 같음
                //순전파 : 1*m(입력)*m*n(가중치)=1*n(출력)
                //역전파 : 1*n(오차)*n*m(가중치 전치)=1*n(내 오차)
    }
    

    //Adam 가중치 업데이트
    public static void updateWeightsAdam(double[][] weights, double[][] dL_dw,
                                         double[][] m_W, double[][] v_W,
                                         int t, double q, double beta1, double beta2, double epsilon) {

        double correction1 = 1.0 - Math.pow(beta1, t);
        double correction2 = 1.0 - Math.pow(beta2, t);

        for (int i = 0; i < weights.length; i++) {          // i: intputSize (행)
            for (int j = 0; j < weights[0].length; j++) {   // j: outputSize (열)
                
                double g = dL_dw[i][j];//미니배치가 적용된 가중치 기울기 평균

                m_W[i][j] = beta1 * m_W[i][j] + (1 - beta1) * g;
                v_W[i][j] = beta2 * v_W[i][j] + (1 - beta2) * g * g;

                double m_hat = m_W[i][j] / correction1;
                double v_hat = v_W[i][j] / correction2;

                weights[i][j] -= q * m_hat / (Math.sqrt(v_hat) + epsilon);
            }
        }
    }

    //Adam 바이어스 업데이트
    public static void updateBiasesAdam(double[] bias, double[] dL_db, double[] m_b, double[] v_b, 
                                        int t, double q, double beta1, double beta2, double epsilon) {
        
        double correction1 = 1.0 - Math.pow(beta1, t);
        double correction2 = 1.0 - Math.pow(beta2, t);

        for (int i = 0; i < bias.length; i++) {
            double g = dL_db[i];
            
            m_b[i] = beta1 * m_b[i] + (1 - beta1) * g;
            v_b[i] = beta2 * v_b[i] + (1 - beta2) * g * g;

            double m_hat = m_b[i] / correction1;
            double v_hat = v_b[i] / correction2;

            bias[i] -= q * m_hat / (Math.sqrt(v_hat) + epsilon);
        }
    }
}