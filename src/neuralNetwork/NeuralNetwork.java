package neuralNetwork;

import data.DataReader;
import data.Image;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import javax.swing.SwingUtilities;
import layers.HiddenLayer;
import layers.Layer;
import layers.OutputLayer;

public class NeuralNetwork {

    private Layer[] layers; 

    private final List<Image> trainImages = DataReader.readData("C:\\Users\\Chaelin\\Desktop\\자료\\Back Propagation\\MNIST 구현\\DNN\\src\\data\\mnist_train.csv");
    private final List<Image> testImages  = DataReader.readData("C:\\Users\\Chaelin\\Desktop\\자료\\Back Propagation\\MNIST 구현\\DNN\\src\\data\\mnist_test.csv");
    
    private double[] currInput;
    private double[] currOutput;

    private final int inputSize = 784;
    private final int[] hiddenLayers;
    private final int size;
    private final int outputSize = 10;
    private final double LEARNING_RATE;
    
    // 그래프 기록용 리스트
    private final List<Double> historyTrainLoss = new ArrayList<>();
    private final List<Double> historyTestLoss = new ArrayList<>();
    private final List<Double> historyTrainAcc = new ArrayList<>();
    private final List<Double> historyTestAcc = new ArrayList<>();
    
    // 시각화 도구
    private final TrainingVisualizer visualizer;

    public NeuralNetwork(int[] hidden, double learningRate){
        this.hiddenLayers = hidden;
        this.LEARNING_RATE = learningRate;
        size = hiddenLayers.length + 1;
        currInput = new double[inputSize];
        initLayers();
        preprocessData(trainImages);
        preprocessData(testImages);
        visualizer = new TrainingVisualizer();
        visualizer.showWindow();
    }
    //min-max 정규화
    private void preprocessData(List<Image> images) {
        for (Image img : images) {
            double[] data = img.getData();
            for (int i = 0; i < data.length; i++) {
                // 원본 데이터를 덮어씌움 (메모리 절약 + 속도 향상)
                data[i] = data[i] / 255.0; 
            }
        }
    }
    private void initLayers(){
        layers = new Layer[size];
        layers[0] = new HiddenLayer(currInput, hiddenLayers[0]);
        for(int i=1; i<size-1; i++){
            layers[i] = new HiddenLayer(layers[i-1].getOutputs(), hiddenLayers[i]);
        }
        layers[size-1] = new OutputLayer(layers[size-2].getOutputs(), outputSize);
        currOutput = layers[size - 1].getOutputs();
    }

    public void forwardPropagation(){
        layers[0].setInputs(currInput);
        for(int i=0; i<size; i++){
            layers[i].calculateOutput();
        }
    }

    public void backwardPropagation(double[] target){
        OutputLayer o = (OutputLayer) layers[size - 1];
        o.calculateLocalGradients(target);
        double[] nextGradient = o.get_dL_dx();
        
        for(int i=size-2; i>=0; i--){
            HiddenLayer h = (HiddenLayer) layers[i];
            h.calculateLocalGradients(nextGradient);
            nextGradient = layers[i].get_dL_dx();
        }
    }

    
    // Adam은 내부적으로 스케일을 조정하므로 배치 사이즈로 나누면 학습이 너무 느려짐
    private void updateWeightsAndBiases(int t){
        double beta1 = 0.9;
        double beta2 = 0.999;
        double epsilon = 1e-8;
        double q = LEARNING_RATE; 
        
        for(Layer l : layers){
            l.updateWeightandBiasAdam(t, q, beta1, beta2, epsilon);
        }
    }

    private double[] test(List<Image> dataSet) {
        double totalLoss = 0;
        int correctCount = 0;
        OutputLayer o = (OutputLayer) layers[size - 1];
        
        for(Image img : dataSet){
            currInput = img.getData();
            layers[0].setInputs(currInput);
            for(int i=0;i<layers.length;i++){
                layers[i].calculateOutput();
            }
            int label = img.getLabel();
            double[] target = new double[outputSize]; 
            target[label] = 1.0;

            double[] outputs = layers[size - 1].getOutputs();
            totalLoss += o.computeLoss(target);
            if(getPredictedLabel(outputs) == label){
                correctCount++;
            }
        }
        return new double[] { totalLoss / dataSet.size(), (double) correctCount / dataSet.size() };
    }

    private int getPredictedLabel(double[] output) {
        int predicted = 0;
        double maxProb = output[0];
        for (int j = 1; j < output.length; j++) {
            if (output[j] > maxProb) {
                maxProb = output[j];
                predicted = j;
            }
        }
        return predicted;
    }

    // --- 속도 최적화된 train 메서드 ---
    public void train_test(int epochs, int batchSize) {

        long startTime = System.currentTimeMillis();
        int timestep = 0;
        System.out.println("Training Started..."); 

        OutputLayer outputLayer = (OutputLayer) layers[size - 1];

        for (int epoch = 0; epoch < epochs; epoch++) {

            long epochStartTime = System.currentTimeMillis();
            System.out.println("\nEpoch " + (epoch + 1) + " started...");
            Collections.shuffle(trainImages);

            double currentEpochTrainLoss = 0;
            int currentEpochCorrect = 0;

            //전체 데이터를 배치 사이즈만큼 건너뛰며 반복
            for (int i = 0; i < trainImages.size(); i += batchSize) {
                int currentBatchSize = Math.min(batchSize, trainImages.size() - i);
                
                // 1. 기울기 초기화 (배치 시작 전)
                for (Layer l : layers) l.resetGradients();

                // 배치 사이즈만큼 기울기 누적 (Accumulate)
                // 이 안에서는 가중치 업데이트 X
                for (int b = 0; b < currentBatchSize; b++) {
                    Image img = trainImages.get(i + b);
                    int label = img.getLabel();
                    currInput = img.getData();

                    double[] trueOutput = new double[outputSize];
                    trueOutput[label] = 1.0;

                    forwardPropagation();
                    
                    currOutput = layers[size - 1].getOutputs(); 
                    
                    currentEpochTrainLoss += outputLayer.computeLoss(trueOutput);
                    if (getPredictedLabel(currOutput) == label) {
                        currentEpochCorrect++;
                    }

                    backwardPropagation(trueOutput); // 기울기 누적 (+=)
                }
                
                // 2. 가중치 업데이트 (배치 처리가 끝난 후 딱 1번만 수행)
                timestep++;
                updateWeightsAndBiases(timestep);
            }

            double avgTrainLoss = currentEpochTrainLoss / trainImages.size();
            double avgTrainAcc = (double) currentEpochCorrect / trainImages.size();

            double[] testMetrics = test(testImages);
            
            historyTrainLoss.add(avgTrainLoss);
            historyTrainAcc.add(avgTrainAcc);
            historyTestLoss.add(testMetrics[0]);
            historyTestAcc.add(testMetrics[1]);

            long epochEndTime = System.currentTimeMillis();
            double epochTime = (epochEndTime - epochStartTime) / 1000.0;

            System.out.printf("Epoch %d Result -> Train Loss: %.5f, Acc: %.2f%% | Test Loss: %.5f, Acc: %.2f%% | Time: %.2fs\n", 
                    (epoch+1), avgTrainLoss, avgTrainAcc * 100, testMetrics[0], testMetrics[1] * 100, epochTime);

            SwingUtilities.invokeLater(() -> {
                visualizer.updateData(historyTrainLoss, historyTestLoss, historyTrainAcc, historyTestAcc);
            });
        }
        System.out.println("Training Finished.");

        long endTime = System.currentTimeMillis();
        double totalTimeSeconds = (endTime - startTime) / 1000.0;

        double[] finalTestMetrics = test(testImages);
        double finalTrainLoss = historyTrainLoss.get(historyTrainLoss.size() - 1);
        double finalTrainAcc  = historyTrainAcc.get(historyTrainAcc.size() - 1);

        System.out.println("\n" + "=".repeat(40));
        System.out.println("         FINAL TRAINING RESULTS         ");
        System.out.println("=".repeat(40));
        
        System.out.printf("Final Train Loss     : %.5f\n", finalTrainLoss);
        System.out.printf("Final Train Accuracy : %.2f%%\n", finalTrainAcc * 100.0);
        System.out.println("-".repeat(40));
        
        System.out.printf("Final Test Loss      : %.5f\n", finalTestMetrics[0]);
        System.out.printf("Final Test Accuracy  : %.2f%%\n", finalTestMetrics[1] * 100.0);
        System.out.println("=".repeat(40));

        System.out.printf("Total Training Time  : %.3f seconds\n", totalTimeSeconds);
        System.out.println("=".repeat(40));
    }
}