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

        dL_dz = new double[outputSize];//해당 노드의 오차 책임, 델타
        dL_dx = new double[inputSize];//델타*가중치 = 이전 층 노드로 전달할 오차 -> 이전 층의 dL_da

        dL_dw = new double[ inputSize ][ outputSize ];//업데이트 할 가중치에 쓸 공간 == edge 개수
        dL_db = new double[ outputSize ];

        m_W = new double[ inputSize ][ outputSize ];
        v_W = new double[ inputSize ][ outputSize ];
        m_b = new double[outputSize];
        v_b = new double[outputSize];

        initWeights();
        initBiases();
    }

    private void initWeights(){//Xavier + He 초기화
        weights = new double[inputSize][outputSize];
        double stddev = 0.01;
        if(this instanceof OutputLayer){
            stddev = Math.sqrt(2.0/(inputSize + outputSize));// Xavier - 출력층의 Softmax 활성화 함수에 적합
        }
        else{
            stddev = Math.sqrt(2.0 / inputSize); // He 초기화 - 은닉층의 ReLU 활성화 함수에 적합 -> 0이하에서 데이터가 죽기 때문에 가중치를 2배 더 크게 설정해서 상쇄
        }
        for ( int i = 0; i < inputSize; i++ )
            for ( int j = 0; j < outputSize; j++ )
                weights[ i ][ j ] = rand.nextGaussian() * stddev;//평균이 0이고 표준편차가 1인 정규분포에서 랜덤한 숫자를 하나 뽑음, stdev(범위) 곱함
    }

    private void initBiases(){
        biases = new double[outputSize];
        Arrays.fill(biases, 0.0);
    }//바이어스는 처음엔 0으로 설정, 바이어스는 그래프를 위아래로 이동시키는 역할
    
    protected abstract void activation(double[] preActOutputs, double[] actOutputs);//층마다 적용시키는 활성화 함수가 다르기 때문에 추상클래스로 정의, 하위 클래스에서 구체화

    public void calculateOutput(){// z=Wx + b
        Arrays.fill(preActOutputs, 0);//초기화
        Operations.calculateWeightedSum(inputs, weights, preActOutputs);//W*x
        for(int i=0; i<outputSize; i++){
            preActOutputs[i] += biases[i];
        }//W*x+b
        activation(preActOutputs, actOutputs);//활성화 함수 적용
    }//노드를 통과한 후 최종 output 

    public void calculateLocalGradients(){//각종 기울기 계산 : dL_dw, dL_db, dL_dx
        //dL_dw 계산
        Operations.calculateWeightGradient(inputs, dL_dz, dL_dw);
        //dL_db 계산
        for(int i=0; i<outputSize; i++){
            dL_db[i] += dL_dz[i];
        }
        //dL_dx 계산
        Operations.calculatePrevLayerError(dL_dz, weights, dL_dx);

    }

   public void resetGradients()
    {
        Arrays.fill( dL_dz, 0.0 );
        Arrays.fill( dL_dx, 0.0 );


        for ( int i = 0; i < inputSize; i++ )
            
            Arrays.fill( dL_dw[ i ], 0.0 );
        
        Arrays.fill( dL_db, 0.0 );
    }// 다른 데이터를 넣기 전에 이전 데이터로 구한 기울기들을 0으로 초기화

    public void setInputs(double[] inputs){
        this.inputs = inputs; 
    }//해당 층의 input 설정
    public double[] getOutputs(){
        return actOutputs;
    }// 출력값을 반환해 다음 입력값으로 설정하기 위해
    

    public double[] get_dL_dx(){
        return dL_dx;
    }//현재 층에서 구한 오차를 다음 층에 넘김, dL_da가 됨
    
    public void updateWeightandBiasAdam(int t, double q, double beta1, double beta2, double epsilon, int batchSize){
        //가중치 기울기 평균 내기
        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                dL_dw[i][j] /= batchSize; // 하나씩 꺼내서 나누기
            }
        }

        //바이어스 기울기 평균 내기
        for (int i = 0; i < outputSize; i++) {
            dL_db[i] /= batchSize; // 하나씩 꺼내서 나누기
        }

        Operations.updateWeightsAdam(weights, dL_dw, m_W, v_W, t, q, beta1, beta2, epsilon);
        Operations.updateBiasesAdam(biases, dL_db, m_b, v_b, t, q, beta1, beta2, epsilon);
        
    }// 학습 후 가중치, 바이어스 업데이트 -> adam 최적화

}
