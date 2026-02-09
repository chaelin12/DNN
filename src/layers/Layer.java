package layers;

import java.util.*;
public abstract class Layer {
    private double[] inputs;
    private double[][] weights;
    private double[] biases;
    protected double[] preActOutputs;
    protected double[] actOutputs;

    protected int inputSize;
    protected int outputSize;

    protected double[] dL_dz;
    protected double[] dL_dx;

    protected double[][] dL_dw;
    protected double[] dL_db;

    public double[][] m_W, v_W;
    public double[] m_b, v_b;
    Random rand = new Random();

    public Layer(double[] input, int outputSize){

        rand = new Random();
        this.inputs = input;
        this.inputSize = input.length;
        this.outputSize = outputSize;


        preActOutputs = new double[outputSize];
        actOutputs = new double[outputSize];

        dL_dz = new double[outputSize];
        dL_dx = new double[inputSize];

        dL_dw = new double[ inputSize ][ outputSize ];//업데이트 할 가중치에 쓸 공간 == edge 개수
        dL_db = new double[ outputSize ];

        m_W = new double[ inputSize ][ outputSize ];
        v_W = new double[ inputSize ][ outputSize ];
        m_b = new double[outputSize];
        v_b = new double[outputSize];

        initWeights();
        initBiases();
    }

    private void initWeights(){//Xavier Initialization
        weights = new double[inputSize][outputSize];
        double stddev = 0.01;
        if(this instanceof OutputLayer){
            stddev = Math.sqrt(2.0/(inputSize + outputSize));// Xavier - 출력층의 Softmax 활성화 함수에 적합
        }
        else{
            stddev = Math.sqrt(2.0 / inputSize); // He 초기화 - 은닉층의 ReLU 활성화 함수에 적합
        }
        for ( int i = 0; i < inputSize; i++ )

            for ( int j = 0; j < outputSize; j++ )

                weights[ i ][ j ] = rand.nextGaussian() * stddev;//평균이 0이고 표준편차가 1인 정규분포에서 랜덤한 숫자를 하나 뽑음, stdev(범위) 곱함
        
    }

    private void initBiases(){
        biases = new double[outputSize];
        Arrays.fill(biases, 0.0);
    }
    
    protected abstract void activation(double[] preActOutputs, double[] actOutputs);

    public void calculateOutput(){// z=Wx + b
        Arrays.fill(preActOutputs, 0);
        Operations.computeWeightedSum(inputs, weights, preActOutputs);
        for(int i=0; i<outputSize; i++){
            preActOutputs[i] += biases[i];
        }
        activation(preActOutputs, actOutputs);
    }

    public void calculateLocalGradients(){
        //dL_dw 계산
        Operations.computeWeightGradient(inputs, dL_dz, dL_dw);
        //dL_db 계산
        for(int i=0; i<outputSize; i++){
            dL_db[i] += dL_dz[i];
        }
        //dL_dx 계산
        Operations.computePrevLayerError(dL_dz, weights, dL_dx);

    }

   public void resetGradients()   // resetting gradients back to zero 
    {
        Arrays.fill( dL_dz, 0.0 );
        Arrays.fill( dL_dx, 0.0 );


        for ( int i = 0; i < inputSize; i++ )
            
            Arrays.fill( dL_dw[ i ], 0.0 );
        

        Arrays.fill( dL_db, 0.0 );
    }// 다른 데이터를 넣기 전에 이전 데이터로 구한 기울기들을 0으로 초기화

    public void setInputs(double[] inputs){
    // 단순히 참조만 연결하는 것이 가장 빠르고 안전합니다.
        this.inputs = inputs; 
    }
    public double[] getOutputs(){
        return actOutputs;
    }// 출력값을 반환해 다음 입력값으로 설정하기 위해

    public double[] get_dL_dx(){
        return dL_dx;
    }
    
    public void updateWeightandBiasAdam(int t, double q, double beta1, double beta2, double epsilon){
        Operations.updateWeightsAdam(weights, dL_dw, m_W, v_W, t, q, beta1, beta2, epsilon);
        Operations.updateBiasesAdam(biases, dL_db, m_b, v_b, t, q, beta1, beta2, epsilon);
        
    }


    


}
