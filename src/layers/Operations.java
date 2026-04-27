package layers;

import java.util.*;

public class Operations {


    public static void calculateWeightedSum(double[] inputs, double[][] weights, double[] output) {
        int m = inputs.length;
        int w = weights.length;
        int n = weights[0].length;

        Arrays.fill(output, 0.0);

        if ( m != w )
        {
            System.err.println("행렬 곱 연산 불가");
            return;
        }

        for (int i = 0; i < m; i++) {
            if (inputs[i] == 0) continue;
            double inputVal = inputs[i];
            for (int j = 0; j < n; j++) {
                output[j] += inputVal * weights[i][j];
            }
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

    //가중치 기울기 계산: dL_dw += dL_dz * Input^T
    public static void calculateWeightGradient(double[] input, double[] dL_dz, double[][] dL_dw){
        int m = input.length;
        int n = dL_dz.length;

        for ( int i = 0; i < m; i++ )
            for ( int j = 0; j < n; j++ )
                dL_dw[ i ][ j ] += input[ i ] * dL_dz[ j ]; 
    
    }


    //이전 층 오차 전파: dL_dx += Weights^T * dL_dz
    public static void calculatePrevLayerError(double[] dL_dz, double[][] weights, double[] dL_dx){
        int n = dL_dz.length;
        int m = weights.length;
        int nn = weights[0].length;
        if ( n != nn )
        {
            System.err.println("행렬 곱 연산 불가");
            return;
        }
        Arrays.fill( dL_dx, 0.0 );
        for ( int i = 0; i < m; i++ )
            for ( int j = 0; j < n; j++ ) 
                dL_dx[ i ] += dL_dz[ j ] * weights[ i ][ j ];
    }
    

    //Adam 가중치 업데이트
    public static void updateWeightsAdam(double[][] weights, double[][] dL_dw, double[][] m_W, double[][] v_W, int t, double q, double beta1, double beta2, double epsilon){
        double correction1 = 1.0 - Math.pow(beta1, t);
        double correction2 = 1.0 - Math.pow(beta2, t);
        for (int i = 0; i < weights.length; i++) { 
            for (int j = 0; j < weights[0].length; j++) {
                double g = dL_dw[i][j];
                m_W[i][j] = beta1 * m_W[i][j] + (1 - beta1) * g;
                v_W[i][j] = beta2 * v_W[i][j] + (1 - beta2) * g * g;
                double m_hat = m_W[i][j] / correction1;
                double v_hat = v_W[i][j] / correction2;
                weights[i][j] -= q * m_hat / (Math.sqrt(v_hat) + epsilon);
            }
        }
    }

    //Adam 바이어스 업데이트
    public static void updateBiasesAdam(double[] bias, double[] dL_db, double[] m_b, double[] v_b, int t, double q, double beta1, double beta2, double epsilon){
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