package neuralNetwork;

import data.Image;
import data.M_DataReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import javax.swing.*;
import layers.HiddenLayer;
import layers.Layer;
import layers.OutputLayer;

public class NeuralNetwork {

    private Layer[] layers; 

    List<Image> trainImages = M_DataReader.readData("C:\\Users\\Chaelin\\Desktop\\자료\\Back Propagation\\MNIST 구현\\DNN\\src\\data\\mnist\\mnist_train.csv");
    List<Image> testImages = M_DataReader.readData("C:\\Users\\Chaelin\\Desktop\\자료\\Back Propagation\\MNIST 구현\\DNN\\src\\data\\mnist\\mnist_test.csv");

    private double[] currInput;
    private double[] currOutput;

    private final int inputSize = 784;
    private final int[] hiddenLayers;
    private final int size;
    private final int outputSize = 10;
    private final double LEARNING_RATE;
    
    private final List<Double> historyTrainLoss = new ArrayList<>();
    private final List<Double> historyTestLoss = new ArrayList<>();
    private final List<Double> historyTrainAcc = new ArrayList<>();
    private final List<Double> historyTestAcc = new ArrayList<>();
    
    private final TrainingVisualizer visualizer;

   
    private final int PATIENCE = 10;
    private int patienceCounter = 0;
    private double bestTestLoss = Double.MAX_VALUE;
    private int bestEpoch = 0;
    private int overfitEpoch = -1;
    
    private List<double[][]> bestWeights;
    private List<double[]> bestBiases;


    public NeuralNetwork(int[] hidden, double learningRate){
        this.hiddenLayers = hidden;
        this.LEARNING_RATE = learningRate;
        size = hiddenLayers.length + 1;
        currInput = new double[inputSize];
        initLayers();
        
        visualizer = new TrainingVisualizer();
        visualizer.showWindow();
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

    private void updateWeightsAndBiases(int t, int b){ 
        double beta1 = 0.9;
        double beta2 = 0.999;
        double epsilon = 1e-8;
        double q = LEARNING_RATE; 
        for(Layer l : layers){
            l.updateWeightandBiasAdam(t, q, beta1, beta2, epsilon, b);
        }
    }
    
    private double[] test(List<Image> dataSet) {
        double totalLoss = 0;
        int correctCount = 0;
        OutputLayer o = (OutputLayer) layers[size - 1];

        for (Image img : dataSet) {
            currInput = img.getData();
            forwardPropagation();
            int label = img.getLabel();
            double[] target = new double[outputSize];
            target[label] = 1.0;
            double[] outputs = layers[size - 1].getOutputs();
            totalLoss += o.calculateLoss(target);

            int predicted = getPredictedLabel(outputs);

            if (predicted == label) {
                correctCount++;
            } 
        }
        return new double[]{totalLoss / dataSet.size(), (double) correctCount / dataSet.size()};
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

    public void train_test(int epochs, int batchSize) {
        visualizer.setMaxEpochs(epochs);

        long startTime = System.currentTimeMillis();
        int timestep = 0;
        double[] initialTrain = test(trainImages); 
        double[] initialTest = test(testImages);

        historyTrainLoss.add(initialTrain[0]);
        historyTrainAcc.add(initialTrain[1]);
        historyTestLoss.add(initialTest[0]);
        historyTestAcc.add(initialTest[1]);

        List<TrainingVisualizer.ImageSample> initialSamples = generateVisualizationSamples();
   
        SwingUtilities.invokeLater(() -> {
            visualizer.updateData(historyTrainLoss, historyTestLoss, historyTrainAcc, historyTestAcc, initialSamples);
        });
        
        System.out.println("\n학습 시작>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"); 
        OutputLayer outputLayer = (OutputLayer) layers[size - 1];

        boolean shouldStop = false; // Early stopping 플래그

        for (int epoch = 0; epoch < epochs; epoch++) {
            long epochStartTime = System.currentTimeMillis();
            System.out.println("\n------------------------------------------Epoch " + (epoch + 1) + "------------------------------------------");
            Collections.shuffle(trainImages);

            double currentEpochTrainLoss = 0;
            int currentEpochCorrect = 0;

            for (int i = 0; i < trainImages.size(); i += batchSize) {
                int currentBatchSize = Math.min(batchSize, trainImages.size() - i);
                
                for (Layer l : layers) l.resetGradients();

                for (int b = 0; b < currentBatchSize; b++) {
                    Image img = trainImages.get(i + b);
                    int label = img.getLabel();
                    currInput = img.getData();

                    double[] target = new double[outputSize];
                    target[label] = 1.0;

                    forwardPropagation();
                    currOutput = layers[size - 1].getOutputs(); 
                    
                    currentEpochTrainLoss += outputLayer.calculateLoss(target);
                    if (getPredictedLabel(currOutput) == label) {
                        currentEpochCorrect++;
                    }
                    backwardPropagation(target);
                }
                
                timestep++;
                updateWeightsAndBiases(timestep, currentBatchSize);
            }

            double avgTrainLoss = currentEpochTrainLoss / trainImages.size();
            double avgTrainAcc = (double) currentEpochCorrect / trainImages.size();
            double[] test = test(testImages);
            double currentTestLoss = test[0];
            
            historyTrainLoss.add(avgTrainLoss);
            historyTrainAcc.add(avgTrainAcc);
            historyTestLoss.add(currentTestLoss);
            historyTestAcc.add(test[1]);

            int currentIndex = epoch + 1; 
            
            if (overfitEpoch == -1) {
                if (currentTestLoss < bestTestLoss) {
                    bestTestLoss = currentTestLoss;
                    bestEpoch = currentIndex;
                    patienceCounter = 0;
                    saveBestModel();
                } else {
                    patienceCounter++;
                    if (patienceCounter >= PATIENCE) {
                        overfitEpoch = currentIndex;
                        System.out.printf("\n[Early Stopping] Epoch %d에서 조기 종료 (Best: Epoch %d)\n", 
                            currentIndex, bestEpoch);
                        
                        int finalBest = bestEpoch;
                        int finalOverfit = overfitEpoch;
                        SwingUtilities.invokeLater(() -> {
                            visualizer.setMarkers(finalBest, finalOverfit);
                        });
                        
                        shouldStop = true;
                    }
                }
            }
        
            List<TrainingVisualizer.ImageSample> samples = generateVisualizationSamples();

            long epochEndTime = System.currentTimeMillis();
            double epochTime = (epochEndTime - epochStartTime) / 1000.0;

            System.out.printf("\nResult -> Train Loss: %.5f, Acc: %.2f%% | Test Loss: %.5f, Acc: %.2f%% | Time: %.2fs\n", 
                     avgTrainLoss, avgTrainAcc * 100, currentTestLoss, test[1] * 100, epochTime);

            SwingUtilities.invokeLater(() -> {
                visualizer.updateData(historyTrainLoss, historyTestLoss, historyTrainAcc, historyTestAcc, samples);
            });

            if (shouldStop) break; // 해당 epoch 출력/시각화 완료 후 종료
            
        } 
        
        System.out.println("\nTraining Finished.");
        long endTime = System.currentTimeMillis();
        double totalTimeSeconds = (endTime - startTime) / 1000.0;

        int finalBest = bestEpoch;
        int finalOverfit = overfitEpoch;
        SwingUtilities.invokeLater(() -> {
            visualizer.setMarkers(finalBest, finalOverfit);
        });

        restoreBestModel();
        double[] finalTestMetrics = test(testImages);

        System.out.println("\n" + "=".repeat(40));
        System.out.println("         FINAL TRAINING RESULTS         ");
        System.out.println("=".repeat(40));
        System.out.printf("Best Model Epoch     : Epoch %d\n", bestEpoch);
        if (overfitEpoch != -1) {
            System.out.printf("Overfit Started At   : Epoch %d\n", overfitEpoch);
        }
        System.out.println("-".repeat(40));
        
        System.out.printf("Final Test Loss      : %.5f\n", finalTestMetrics[0]);
        System.out.printf("Final Test Accuracy  : %.2f%%\n", finalTestMetrics[1] * 100.0);
        System.out.println("=".repeat(40));

        System.out.printf("Total Training Time  : %.3f seconds\n", totalTimeSeconds);
        System.out.println("=".repeat(40));
    }

    private List<TrainingVisualizer.ImageSample> generateVisualizationSamples() {
        List<TrainingVisualizer.ImageSample> samples = new ArrayList<>();
        Collections.shuffle(testImages);
        
        int sampleSize = Math.min(10, testImages.size());
        
        for(int i=0; i<sampleSize; i++) {
            Image img = testImages.get(i);
            
            currInput = img.getData();
            forwardPropagation();
            
            double[] out = layers[size-1].getOutputs();
            int predicted = getPredictedLabel(out);
            double[] pixelCopy = img.getData().clone();
            
            samples.add(new TrainingVisualizer.ImageSample(pixelCopy, img.getLabel(), predicted));
        }
        return samples;
    }

    private void saveBestModel() {
        bestWeights = new ArrayList<>();
        bestBiases = new ArrayList<>();
        
        for (Layer l : layers) {
            double[][] w = l.getWeights();
            double[] b = l.getBiases();

            double[][] wCopy = new double[w.length][w[0].length];
            for (int i = 0; i < w.length; i++) {
                wCopy[i] = w[i].clone();
            }
            
            bestWeights.add(wCopy);
            bestBiases.add(b.clone());
        }
    }

    private void restoreBestModel() {
        if (bestWeights == null || bestWeights.isEmpty()) return;
        
        for (int i = 0; i < layers.length; i++) {
            layers[i].setWeights(bestWeights.get(i));
            layers[i].setBiases(bestBiases.get(i));
        }
    }

}