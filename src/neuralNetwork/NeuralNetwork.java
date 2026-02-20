package neuralNetwork;

import data.DataReader;
import data.Image;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import javax.swing.*;
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
        long startTime = System.currentTimeMillis();
        int timestep = 0;

        System.out.println("\n학습 시작>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"); 
        OutputLayer outputLayer = (OutputLayer) layers[size - 1];

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
            
            historyTrainLoss.add(avgTrainLoss);
            historyTrainAcc.add(avgTrainAcc);
            historyTestLoss.add(test[0]);
            historyTestAcc.add(test[1]);
            
            List<TrainingVisualizer.ImageSample> samples = generateVisualizationSamples();

            long epochEndTime = System.currentTimeMillis();
            double epochTime = (epochEndTime - epochStartTime) / 1000.0;

            System.out.printf("\nResult -> Train Loss: %.5f, Acc: %.2f%% | Test Loss: %.5f, Acc: %.2f%% | Time: %.2fs\n", 
                     avgTrainLoss, avgTrainAcc * 100, test[0], test[1] * 100, epochTime);

            SwingUtilities.invokeLater(() -> {
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

        // *** 학습 완료 후 단일 랜덤 이미지 상세 분석 창 띄우기 ***
        SwingUtilities.invokeLater(this::showDetailedPrediction);
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

    // *** 세로형 열 벡터(Column Vector) 디자인이 적용된 상세 분석 창 ***
    private void showDetailedPrediction() {
        java.util.Random rand = new java.util.Random();
        data.Image img = testImages.get(rand.nextInt(testImages.size()));
        
        // 데이터가 전부 0인 빈 이미지가 뽑히는 것을 방지
        while (true) {
            double sum = 0;
            for (double v : img.getData()) sum += v;
            if (sum > 5.0) break;
            img = testImages.get(rand.nextInt(testImages.size()));
        }

        final double[] targetPixels = img.getData().clone();
        currInput = targetPixels;
        int actualLabel = img.getLabel();

        forwardPropagation(); 
        final double[] probs = layers[size - 1].getOutputs().clone();

        // 창 크기를 가로로 더 넓게 조정 (이미지 - 그리드 - 벡터가 나란히 들어가도록)
        JFrame frame = new JFrame("Single Image Detailed Analysis");
        frame.setSize(1350, 700);
        frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        frame.setLocationRelativeTo(null);

        JPanel panel = new JPanel() {
            @Override
            protected void paintComponent(Graphics g) {
                super.paintComponent(g);
                Graphics2D g2 = (Graphics2D) g;
                g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

                g2.setColor(Color.WHITE);
                g2.fillRect(0, 0, getWidth(), getHeight());

                int cellSize = 17; 
                int imgDisplaySize = 28 * cellSize; // 476
                int leftX = 50;
                int middleX = leftX + imgDisplaySize + 50;
                int rightX = middleX + imgDisplaySize + 70;
                int startY = 80;

                // 1. [왼쪽] 원본 이미지
                g2.setColor(Color.BLACK);
                g2.setFont(new Font("SansSerif", Font.BOLD, 18));
                g2.drawString("Original Image (28 x 28)", leftX, startY - 20);
                
                BufferedImage bImg = new BufferedImage(28, 28, BufferedImage.TYPE_INT_RGB);
                for (int r = 0; r < 28; r++) {
                    for (int c = 0; c < 28; c++) {
                        double val = targetPixels[r * 28 + c]; 
                        int gray = (int) (val * 255);
                        if (gray < 0) gray = 0;
                        if (gray > 255) gray = 255;
                        
                        int rgb = (gray << 16) | (gray << 8) | gray;
                        bImg.setRGB(c, r, rgb);
                    }
                }
                g2.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_NEAREST_NEIGHBOR);
                g2.drawImage(bImg, leftX, startY, imgDisplaySize, imgDisplaySize, null);
                g2.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BICUBIC);
                g2.setColor(Color.BLACK);
                g2.drawRect(leftX, startY, imgDisplaySize, imgDisplaySize); 

                // 2. [가운데] 정규화(Min-Max) 수치 그리드
                g2.setFont(new Font("SansSerif", Font.BOLD, 18));
                g2.drawString("Normalized Values Grid", middleX, startY - 20);

                g2.setFont(new Font("SansSerif", Font.PLAIN, 8)); 
                java.text.DecimalFormat df = new java.text.DecimalFormat("0.##"); 

                for (int r = 0; r < 28; r++) {
                    for (int c = 0; c < 28; c++) {
                        double val = targetPixels[r * 28 + c];
                        int cx = middleX + c * cellSize;
                        int cy = startY + r * cellSize;

                        g2.setColor(new Color(220, 220, 220)); 
                        g2.drawRect(cx, cy, cellSize, cellSize);

                        g2.setColor(Color.BLACK);
                        String text = (val < 0.01) ? "0" : df.format(val);
                        FontMetrics fm = g2.getFontMetrics();
                        int tx = cx + (cellSize - fm.stringWidth(text)) / 2;
                        int ty = cy + (cellSize - fm.getHeight()) / 2 + fm.getAscent();
                        g2.drawString(text, tx, ty);
                    }
                }
                g2.setColor(Color.BLACK);
                g2.drawRect(middleX, startY, imgDisplaySize, imgDisplaySize); 

                // 3. [오른쪽] Softmax 예측 벡터 (요청하신 디자인)
                int boxW = 160;
                int rowH = 40;
                int boxH = rowH * 10;
                int boxY = startY + (imgDisplaySize - boxH) / 2; // 세로 중앙 정렬

                // "P (예측 값)" 타이틀
                g2.setFont(new Font("SansSerif", Font.BOLD, 22));
                g2.setColor(new Color(0, 32, 96)); // 짙은 네이비
                String titleText = "P (예측 값)";
                FontMetrics fmTitle = g2.getFontMetrics();
                g2.drawString(titleText, rightX + (boxW - fmTitle.stringWidth(titleText))/2, boxY - 20);

                // 하이라이트 배경 칠하기 (정답 레이블 기준)
                for (int i = 0; i < 10; i++) {
                    if (i == actualLabel) {
                        g2.setColor(new Color(255, 230, 204)); // 요청하신 살구색
                        g2.fillRect(rightX, boxY + i * rowH, boxW, rowH);
                    }
                }

                // 박스 테두리 (두껍게)
                g2.setColor(new Color(0, 32, 96));
                g2.setStroke(new BasicStroke(3f));
                g2.drawRect(rightX, boxY, boxW, boxH);
                g2.setStroke(new BasicStroke(1f));

                // 소수점 8자리 숫자 출력
                g2.setFont(new Font("SansSerif", Font.PLAIN, 18));
                FontMetrics fmVal = g2.getFontMetrics();
                for (int i = 0; i < 10; i++) {
                    g2.setColor(Color.BLACK);
                    // 소수점 8자리 포맷팅
                    String valText = String.format("%.8f", probs[i]);
                    
                    int tx = rightX + (boxW - fmVal.stringWidth(valText)) / 2;
                    int ty = boxY + i * rowH + (rowH - fmVal.getHeight()) / 2 + fmVal.getAscent();
                    g2.drawString(valText, tx, ty);
                }
                
                // 직관성을 위해 박스 우측에 정답 라벨 화살표 추가
                g2.setColor(Color.GRAY);
                g2.setFont(new Font("SansSerif", Font.BOLD, 14));
                g2.drawString("<- Actual: " + actualLabel, rightX + boxW + 15, boxY + actualLabel * rowH + rowH/2 + 5);
            }
        };

        frame.add(panel);
        frame.setVisible(true);
    }
}