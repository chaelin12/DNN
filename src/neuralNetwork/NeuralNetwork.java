package neuralNetwork;

import layers.*;
import java.util.*;
import javax.swing.*;
import java.awt.*;
import java.util.List;               // List 충돌 방지
import java.awt.geom.AffineTransform; // 글자 회전용
import data.dataReader;
import data.Image;


public class NeuralNetwork {

    private TrainingChart chart;
    private Layer[] layers; // 생성된 층들을 저장하는 배열
    private List< Image > images; // 학습에 사용할 이미지 데이터
    private double[] currInput;
    private double[] currOutput;

    private final int inputSize = 2;
    private final int[] hiddenLayers;
    private final int size;
    private final int OutputSize = 1;
    private final double LEARNING_RATE;

    public NeuralNetwork(int[] hidden, double learningRate){
        this.hiddenLayers = hidden;
        this.LEARNING_RATE = learningRate;
        this.size = hiddenLayers.length + 1; //은닉층 + 출력층
        currInput = new double[inputSize];
        InitLayers();
        this.chart = new TrainingChart("XOR Training Result");
    }
    
    public void InitLayers(){
        layers = new Layer[size];
        //첫 번째 은닉층
        layers[0] = new HiddenLayer(currInput, hiddenLayers[0]);

        //나머지 은닉층
        for(int i=1; i<size-1; i++){
            layers[i] = new HiddenLayer(layers[i-1].getOutputs(), hiddenLayers[i]);
        }

        //출력층
        layers[size-1] = new OutputLayer(layers[size-2].getOutputs(), OutputSize);
    }

    public void forwardPropagation(){
        layers[0].setInputs(currInput);
        for(int i=1; i<size; i++){
            layers[i].calculateOutput();
        }
    }

    public void backwardPropagation(double[] target){
        for(int i=0; i<size; i++){
            layers[i].resetGradients();
        }
        //출력층의 기울기 계산 -> dL_dx
        ((OutputLayer)layers[size-1]).calculateLocalGradients(target);

        double[] nextGradient = layers[size-1].get_dL_dx();// 출력층의 기울기 가져오기
        
        //은닉층의 기울기 계산
        for(int i=size-2; i>=0; i--){
            ((HiddenLayer)layers[i]).calculateLocalGradients(nextGradient);//현재 은닉층의 기울기 계산
            nextGradient = layers[i].get_dL_dx();//다음 층으로 전달할 기울기 갱신
        }
    }

    private void updateWeightsAndBiases(int t){
        double beta1 = 0.9;
        double beta2 = 0.999;
        double epsilon = 1e-8;

        for(int i=0; i<size; i++){
            layers[i].updateWeightandBiasAdam(t, LEARNING_RATE, beta1, beta2, epsilon);
        }


    public void train(int epochs){

        images = dataReader.readData("C:\\Users\\Chaelin\\Desktop\\자료\\Back Propagation\\MNIST 구현\\DNN\\src\\data\\mnist_train.csv");
        
        int timestep = 0;

        for(int epoch=0;epoch<epochs; epoch++){
            System.out.println("Epoch " + (epoch+1));
            Collections.shuffle(images);//데이터 섞기
            for ( int i = 0; i < images.size(); i++ ) 
            {
                timeStep++;
                
                System.out.println( "Training sample: " + i );

                int label = images.get( i ).getLabel();         // True label (0-9)
                currInput = images.get( i ).getData();          // Input to the network
        
                double[] trueOutput = new double[ OutputSize ]; // hot coded vector (arr[trueLabel] =  1 else 0)
                trueOutput[ label ] = 1;
                
                forwardPropagation();               // get output for each input

                backwardPropagation( trueOutput ); // get error(Loss->scalar), compute gradients (calculate by how much the network is wrong)
                
                updateWeightsAndBiases(timeStep);    // update the parameters based on the the Loss by gradient decent algorithm 
            }
        }
    }
    
}
    