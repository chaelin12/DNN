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
        this.inputs = input;
        this.inputSize = input.length;
        this.outputSize = outputSize;

        this.weights = new double[outputSize][inputSize];
        this.biases = new double[outputSize];

        this.preActOutputs = new double[outputSize];
        this.actOutputs = new double[outputSize];

        this.dL_dz = new double[outputSize];
        this.dL_dx = new double[inputSize];

        this.dL_dw = new double[outputSize][inputSize];
        this.dL_db = new double[outputSize];

        this.m_W = new double[outputSize][inputSize];
        this.v_W = new double[outputSize][inputSize];
        this.m_b = new double[outputSize];
        this.v_b = new double[outputSize];

        initWeights();
        initBiases();
    }

    private void initWeights(){//Xavier Initialization
        weights = new double[outputSize][inputSize];
        double stddev = 0.01;
        if(this.getClass() == OutputLayer.class){
            stddev = Math.sqrt(6/(inputSize + outputSize));// Xavier - 출력층의 Softmax 활성화 함수에 적합
        }
        else{
            stddev = Math.sqrt(2.0 / inputSize); // He 초기화 - 은닉층의 ReLU 활성화 함수에 적합
        }
        for(int i=0; i<outputSize; i++){
            for(int j=0; j<inputSize; j++){
                weights[i][j] = rand.nextGaussian() * stddev;
            }
        }
        System.out.println("Weight Initialization Complete");
    }

    private void initBiases(){
        for(int i=0; i<outputSize; i++){
            biases[i] = 0.0;
        }
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
            dL_db[i] = dL_dz[i];
        }
        //dL_dx 계산
        Operations.computePrevLayerError(dL_dz, weights, dL_dx);

    }

    public void resetGradients(){
        for(int i=0; i<outputSize; i++){
            dL_db[i] = 0.0;
            for(int j=0; j<inputSize; j++){
                dL_dw[i][j] = 0.0;
            }
        }
    }

    public void setInputs(double[] inputs){
        this.inputs = inputs;}
    
    public double[] getOutputs(){
        return actOutputs;}

    public double[] get_dL_dx(){
        return dL_dx;}
    
    public void updateWeightandBiasAdam(int t, double q, double beta1, double beta2, double epsilon){
        Operations.updateWeightsAdam(weights, dL_dw, m_W, v_W, t, q, beta1, beta2, epsilon);
        Operations.updateBiasesAdam(biases, dL_db, m_b, v_b, t, q, beta1, beta2, epsilon);
    }


    


}
