package layers;

import java.util.*;

public class Operations{
    //행렬 연산 유틸리티 클래스

    public static void computeWeightedSum(double[] inputs, double[][] weights, double[] output){

        int m = inputs.length; //inputs size
        int w = weights.length;//행렬의 행 개수
        int n = weights[0].length; //output size

        if(m != w){// 행렬의 행 개수와 입력 크기가 같지 않으면 행렬 곱셈 불가
            System.out.println("Error in matrix multiplication: incompatible size");
            return;
        }

        for(int j=0; j<n; j++){//출력 끝까지
            for(int i=0; i<m; i++){//입력 크기 만큼 ( m=w )
                output[j] += inputs[i] * weights[i][j];//가중치 곱한 값 더하기
            }
        }
    }//weight*inputs의 합

    //델타 계산 : dL/dz = dL/da * da/dz
    public static void computeDelta(double[] dL_da, double[] da_dz, double[] dL_dz){
        int n1 = dL_da.length;
        int n2 = da_dz.length;
        int n3 = dL_dz.length;

        if ( n1 != n2 && n1 != n3 && n2 != n3 )//연쇄 법칙을 하기 위해서는 모두 길이가 같아야 함
        {
            System.err.println("inputs have different length");
            return;
        }

        for(int i=0; i<n1; i++){
            dL_dz[i] = dL_da[i] * da_dz[i];//연쇄 법칙 적용
        }

    }

    //dL_dW 계산 : dL/dW = input * dL/dz, 업데이트 할 가중치 계산
    public static void computeWeightGradient(double[] input, double[] dL_dz, double[][] dL_dW){
        int m= input.length; //input size
        int n= dL_dz.length; //output size 

        for(int i=0;i<m;i++){//입력 노드 m개
            for(int j=0;j<n;j++){//에 대한 각 오차 책임
                dL_dW[i][j] = input[i] * dL_dz[j];
            }
        }
    }

    //dL_dx 계산 = dL_dz * W^T, 앞 층으로 전달할 오차 계산
    public static void computePrevLayerError(double[] dL_dz, double[][] weights, double[] dL_dx){
        int n = dL_dz.length;//오차의 개수
        int m = weights.length;// 연결된 가중치의 개수 (앞 층의 노드 개수)
        int p = weights[0].length;//가중치의 열 개수(뒷 층의 노드 개수)

        if(n != p){//오차의 개수와 가중치 열 개수가 같아야 함, 오차 개수 == output 개수
            System.err.println("Incompatible size for previous layer error computation");
            return;
        }

        Arrays.fill(dL_dx, 0);//이전 층 오차 초기화

        for(int i=0;i<m;i++){//앞 층의 노드 개수 만큼
            for(int j=0;j<n;j++){//가중치 하나하나씩
                dL_dx[i] += dL_dz[j] * weights[i][j]; // dL_dx 계산
            }
        }
    }
    

    public static void updateWeightsAdam(double[][] weights, double[][] dL_dW,

                                         double[][] m_W, double[][] v_W,

                                         int t, double q, double beta1, double beta2, double epsilon){

        double correction1 = 1.0 - Math.pow(beta1, t);
        double correction2 = 1.0 - Math.pow(beta2, t);//편향 보정 값 계산 : 학습 초반에 m,v 값이 너무 작아 0으로 쏠리는 현상 방지

        for(int i=0; i< weights.length; i++){

            for(int j=0; j< weights[0].length; j++){

                double g = dL_dW[i][j]; //현재 기울기

                m_W[i][j] = beta1 * m_W[i][j] + (1-beta1) * g; //1차 모멘텀 업데이트
                v_W[i][j] = beta2 * v_W[i][j] + (1-beta2) * g * g; //2차 모멘텀 업데이트

                double m_hat = m_W[i][j] / correction1; //편향 보정된 1차 모멘텀
                double v_hat = v_W[i][j] / correction2; //편향 보정된 2차 모멘텀

                weights[i][j] -= q * m_hat / (Math.sqrt(v_hat) + epsilon);
                dL_dW[i][j] = 0; //기울기 초기화

            }
        }
    }


    public static void updateBiasesAdam(double[] bias, double[] dL_db, double[] m_b, double[] v_b, int t, double q, double beta1, double beta2, double epsilon){
        // 편향 보정값(Bias Correction) 미리 계산
        // (학습 초반에 값이 0으로 쏠리는 것을 방지하는 공식) - 분자 계산해서 사용
        double correction1 = 1.0 - Math.pow(beta1, t);// beta1의 t승 : pow
        double correction2 = 1.0 - Math.pow(beta2, t);

        for(int i=0; i< bias.length; i++){

            double g = dL_db[i]; //현재 기울기
                m_b[i] = beta1 * m_b[i] + (1-beta1) * g; //1차 모멘텀 업데이트
                v_b[i] = beta2 * v_b[i] + (1-beta2) * g * g; //2차 모멘텀 업데이트

                double m_hat = m_b[i] / correction1; //편향 보정된 1차 모멘텀
                double v_hat = v_b[i] / correction2; //편향 보정된 2차 모멘텀

                bias[i] -= q * m_hat / (Math.sqrt(v_hat) + epsilon);
                dL_db[i] = 0; //기울기 초기화

            }
        }
    }




    
    