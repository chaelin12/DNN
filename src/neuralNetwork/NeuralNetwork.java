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
    
    private final List<Double> historyTrainLoss = new ArrayList<>();
    private final List<Double> historyTestLoss = new ArrayList<>();
    private final List<Double> historyTrainAcc = new ArrayList<>();
    private final List<Double> historyTestAcc = new ArrayList<>();
    
    private final TrainingVisualizer visualizer;

    public NeuralNetwork(int[] hidden, double learningRate){
        this.hiddenLayers = hidden;
        this.LEARNING_RATE = learningRate;
        size = hiddenLayers.length + 1;
        currInput = new double[inputSize];
        initLayers();
        min_max(trainImages); 
        min_max(testImages);
        
        visualizer = new TrainingVisualizer();
        visualizer.showWindow();
    }


    private void min_max(List<Image> images) {
        for (Image img : images) {
            double[] data = img.getData();
            for (int i = 0; i < data.length; i++) {
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
        layers[0].setInputs(currInput);//입력값 설정
        for(int i=0; i<size; i++){
            layers[i].calculateOutput();//입력값을 통과시킴
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
    }//dL_dz -> dL_dw, dL_dx, dL_db 구함

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
        
        for(Image img : dataSet){
            currInput = img.getData();
            forwardPropagation();
            int label = img.getLabel();
            double[] target = new double[outputSize]; 
            target[label] = 1.0;

            double[] outputs = layers[size - 1].getOutputs();
            totalLoss += o.calculateLoss(target);
            if(getPredictedLabel(outputs) == label){
                correctCount++;
            }
        }

        return new double[] { totalLoss / dataSet.size(), (double) correctCount / dataSet.size() };
    }

    //output 배열 : 각 숫자(0~9)일 확률이 들어있음
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
    }//가장 큰 확률을 가진 숫자의 인덱스를 반환
    

    public void train_test(int epochs, int batchSize) {
        long startTime = System.currentTimeMillis();// 최종 수행 시간 계산
        int timestep = 0;

        System.out.println("\n학습 시작>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"); 
        OutputLayer outputLayer = (OutputLayer) layers[size - 1];

        for (int epoch = 0; epoch < epochs; epoch++) {

            long epochStartTime = System.currentTimeMillis();//epoch 별 수행 시간
            System.out.println("\n------------------------------------------Epoch " + (epoch + 1) + "------------------------------------------");
            Collections.shuffle(trainImages);

            double currentEpochTrainLoss = 0;//epoch별 로스
            int currentEpochCorrect = 0;//epoch별 정답 횟수

            for (int i = 0; i < trainImages.size(); i += batchSize) {// 전체 데이터를 batch사이즈 만큼 건너뛰며 접근
                int currentBatchSize = Math.min(batchSize, trainImages.size() - i);// 배치 사이즈만큼 딱 떨어지지 않을 때
                
                for (Layer l : layers) l.resetGradients();//신경망 통과 전 기울기 초기화

                for (int b = 0; b < currentBatchSize; b++) {//배치(b)의 첫번째 부터 끝까지
                    Image img = trainImages.get(i + b);//학습 이미지의 배치 번째 데이터를 가져옴
                    int label = img.getLabel();//정답
                    currInput = img.getData();//28*28 픽셀의 그림

                    double[] target = new double[outputSize];// 출력 노드 수만큼 정답 배열 생성
                    target[label] = 1.0;//정답 값(=인덱스)에 해당하는 값을 target 배열에서 1로 설정

                    forwardPropagation();
                    currOutput = layers[size - 1].getOutputs(); //순전파 시행 후 나온 output들을 현재 층의 output 배열에 저장
                    
                    currentEpochTrainLoss += outputLayer.calculateLoss(target);//정답 배열과 softmax 결과값으로 오차 계산
                    if (getPredictedLabel(currOutput) == label) {// 가장 큰 확률의 수가 정답값(인덱스)랑 같으면 현재 epoch에서 맞은 수 ++
                        currentEpochCorrect++;
                    }
                    backwardPropagation(target);//역전파 진행
                }
                
                timestep++;//현재 업데이트 수
                updateWeightsAndBiases(timestep,currentBatchSize);
            }

            double avgTrainLoss = currentEpochTrainLoss / trainImages.size();//평균 loss
            double avgTrainAcc = (double) currentEpochCorrect / trainImages.size();//평균 정확률

            double[] test = test(testImages);
            
            historyTrainLoss.add(avgTrainLoss);
            historyTrainAcc.add(avgTrainAcc);
            //학습 후 평균 오차와 정확도 반환
            historyTestLoss.add(test[0]);
            historyTestAcc.add(test[1]);
            //test 후 평균 오차와 정확도 반환
            
            // *** 시각화용 샘플 생성 ***
            List<TrainingVisualizer.ImageSample> samples = generateVisualizationSamples();

            long epochEndTime = System.currentTimeMillis();
            double epochTime = (epochEndTime - epochStartTime) / 1000.0;// epoch가 걸린 시간

            System.out.printf("\nResult -> Train Loss: %.5f, Acc: %.2f%% | Test Loss: %.5f, Acc: %.2f%% | Time: %.2fs\n", 
                     avgTrainLoss, avgTrainAcc * 100, test[0], test[1] * 100, epochTime);

            SwingUtilities.invokeLater(() -> {
                // 샘플 리스트도 함께 전달  
                visualizer.updateData(historyTrainLoss, historyTestLoss, historyTrainAcc, historyTestAcc, samples);
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

    // *** 시각화용 샘플 생성 메서드 추가 ***
    private List<TrainingVisualizer.ImageSample> generateVisualizationSamples() {
        List<TrainingVisualizer.ImageSample> samples = new ArrayList<>();
        Collections.shuffle(testImages); // 테스트 데이터 섞기
        
        // 10개만 뽑아서 예측 수행
        int sampleSize = Math.min(10, testImages.size());
        
        for(int i=0; i<sampleSize; i++) {
            Image img = testImages.get(i);
            
            // 순전파 수행 
            currInput = img.getData();
            forwardPropagation();
            
            double[] out = layers[size-1].getOutputs();
            int predicted = getPredictedLabel(out);
            
            // 원본 데이터 복사 (화면에 그리기 위해)
            double[] pixelCopy = img.getData().clone();
            
            samples.add(new TrainingVisualizer.ImageSample(pixelCopy, img.getLabel(), predicted));
        }
        return samples;
    }
 
}