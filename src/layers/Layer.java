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

        dL_dw = new double[ inputSize ][ outputSize ];
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
            stddev = Math.sqrt(2.0/(inputSize + outputSize));
        }
        else{
            stddev = Math.sqrt(2.0 / inputSize);
        }
        for ( int i = 0; i < inputSize; i++ )
            for ( int j = 0; j < outputSize; j++ )
                weights[ i ][ j ] = rand.nextGaussian() * stddev;
    }

    private void initBiases(){
        biases = new double[outputSize];
        Arrays.fill(biases, 1.0);
    }
    
    protected abstract void activation(double[] preActOutputs, double[] actOutputs);

    public void calculateOutput(){
        Arrays.fill(preActOutputs, 0);
        Operations.calculateWeightedSum(inputs, weights, preActOutputs);
        for(int i=0; i<outputSize; i++){
            preActOutputs[i] += biases[i];
        }
        activation(preActOutputs, actOutputs);
    } 

    public void calculateLocalGradients(){
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
    public double[][] getWeights() { return this.weights; }
    public double[] getBiases() { return this.biases; }

    public void setWeights(double[][] newWeights) { this.weights = newWeights; }
    public void setBiases(double[] newBiases) { this.biases = newBiases; }
    public void updateWeightandBiasAdam(int t, double q, double beta1, double beta2, double epsilon, int batchSize){
        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                dL_dw[i][j] /= batchSize;
            }
        }
        for (int i = 0; i < outputSize; i++) {
            dL_db[i] /= batchSize;
        }
        Operations.updateWeightsAdam(weights, dL_dw, m_W, v_W, t, q, beta1, beta2, epsilon);
        Operations.updateBiasesAdam(biases, dL_db, m_b, v_b, t, q, beta1, beta2, epsilon);
    }

}
