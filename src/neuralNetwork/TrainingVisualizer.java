package neuralNetwork;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Font;
import java.awt.FontMetrics;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.GraphicsEnvironment;
import java.awt.Rectangle;
import java.awt.RenderingHints;
import java.awt.geom.AffineTransform;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;
import javax.swing.JFrame;
import javax.swing.JPanel;

public class TrainingVisualizer extends JPanel {
    private List<Double> trainLoss = new ArrayList<>();
    private List<Double> testLoss = new ArrayList<>();
    private List<Double> trainAcc = new ArrayList<>();
    private List<Double> testAcc = new ArrayList<>();
    
    // *** 이 부분이 반드시 있어야 합니다 ***
    public static class ImageSample {
        public double[] pixels;
        public int actualLabel;
        public int predictedLabel;

        public ImageSample(double[] pixels, int actual, int predicted) {
            this.pixels = pixels;
            this.actualLabel = actual;
            this.predictedLabel = predicted;
        }
    }
    // **********************************
    
    private List<ImageSample> currentSamples = new ArrayList<>();
    private int maxEpochs = 0; 
    private JFrame frame;

    public TrainingVisualizer() {
        trainLoss.add(0.0); testLoss.add(0.0);
        trainAcc.add(0.0); testAcc.add(0.0);
    }

    public void setMaxEpochs(int epochs) {
        this.maxEpochs = epochs;
        repaint();
    }

    public void showWindow() {
        frame = new JFrame("Training Monitor & Prediction Check");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        
        int windowWidth = 1200;
        int windowHeight = 850;
        frame.setSize(windowWidth, windowHeight);
        frame.add(this);

        try {
            GraphicsEnvironment ge = GraphicsEnvironment.getLocalGraphicsEnvironment();
            Rectangle bounds = ge.getMaximumWindowBounds(); 
            int x = bounds.x + bounds.width - windowWidth;
            int y = bounds.y + bounds.height - windowHeight;
            frame.setLocation(x, y);
        } catch (Exception e) {
            frame.setLocationRelativeTo(null); 
        }

        frame.setVisible(true);
    }

    // 데이터 업데이트 메서드
    public void updateData(List<Double> trL, List<Double> teL, List<Double> trA, List<Double> teA, List<ImageSample> samples) {
        this.trainLoss = new ArrayList<>(trL);
        this.testLoss = new ArrayList<>(teL);
        this.trainAcc = new ArrayList<>(trA);
        this.testAcc = new ArrayList<>(teA);
        this.currentSamples = samples;
        repaint();
    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        Graphics2D g2 = (Graphics2D) g;
        
        g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        g2.setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING, RenderingHints.VALUE_TEXT_ANTIALIAS_ON);

        g2.setColor(Color.WHITE);
        g2.fillRect(0, 0, getWidth(), getHeight());

        int w = getWidth();
        int h = getHeight();
        
        int graphHeight = (int)(h * 0.6);
        int graphWidth = w / 2;
        
        drawGraph(g2, 0, 0, graphWidth, graphHeight, 
                  trainLoss, testLoss, 
                  "Train & Test Loss", "Loss", 
                  3.0, 0.6, false);

        drawGraph(g2, graphWidth, 0, graphWidth, graphHeight, 
                  trainAcc, testAcc, 
                  "Train & Test Accuracy", "Accuracy", 
                  1.0, 0.2, true);
        
        drawImageSamples(g2, 0, graphHeight, w, h - graphHeight);
    }
    
    private void drawImageSamples(Graphics2D g2, int x, int y, int w, int h) {
        if (currentSamples == null || currentSamples.isEmpty()) return;

        g2.setColor(Color.LIGHT_GRAY);
        g2.drawLine(x, y, x + w, y); 

        g2.setColor(Color.BLACK);
        g2.setFont(new Font("SansSerif", Font.BOLD, 16));
        g2.drawString("Sample Predictions (Epoch Result)", x + 20, y + 30);

        int sampleCount = currentSamples.size();
        int imgSize = 28; 
        int scale = 4;    
        int displaySize = imgSize * scale; 
        int gap = 30;     
        
        int totalWidth = sampleCount * displaySize + (sampleCount - 1) * gap;
        int startX = (w - totalWidth) / 2;
        int startY = y + 110; 

        for (int i = 0; i < sampleCount; i++) {
            ImageSample sample = currentSamples.get(i);
            int drawX = startX + i * (displaySize + gap);
            
            // 1. 해당 이미지에서 가장 큰 값 찾기 (Contrast 증폭용)
            double maxVal = 0.0;
            for (double v : sample.pixels) {
                if (v > maxVal) maxVal = v;
            }
            if (maxVal <= 0) maxVal = 1.0; // 0으로 나누기 방지

            BufferedImage bImg = new BufferedImage(imgSize, imgSize, BufferedImage.TYPE_INT_RGB);
            
            for (int r = 0; r < imgSize; r++) {
                for (int c = 0; c < imgSize; c++) {
                    double rawVal = sample.pixels[r * imgSize + c];
                    
                    // 2. 최대값을 기준으로 정규화 (0.0 ~ 1.0) -> 가장 진한 부분이 1.0이 됨
                    double normalized = rawVal / maxVal;
                    
                    // 3. 색상 반전: (1.0 - 값) -> 값이 클수록(글씨) 0(검정)에 가까워짐
                    // 배경(0) -> 1.0 -> 255 (흰색)
                    // 글씨(1) -> 0.0 -> 0 (검정)
                    int gray = (int)((1.0 - normalized) * 255);
                    
                    // 범위 안전장치
                    if (gray < 0) gray = 0;
                    if (gray > 255) gray = 255;

                    // 4. ARGB 색상 생성 (불투명)
                    int rgb = (0xFF << 24) | (gray << 16) | (gray << 8) | gray;
                    bImg.setRGB(c, r, rgb);
                }
            }
            
            g2.drawImage(bImg, drawX, startY, displaySize, displaySize, null);
            g2.setColor(Color.BLACK);
            g2.drawRect(drawX, startY, displaySize, displaySize); // 테두리

            // 텍스트 그리기
            String predText = "Pred: " + sample.predictedLabel;
            String actText = "Act: " + sample.actualLabel;
            
            g2.setFont(new Font("SansSerif", Font.BOLD, 14));
            
            if (sample.predictedLabel == sample.actualLabel) {
                g2.setColor(new Color(0, 150, 0)); 
            } else {
                g2.setColor(Color.RED); 
            }
            
            FontMetrics fm = g2.getFontMetrics();
            int tx = drawX + (displaySize - fm.stringWidth(predText)) / 2;
            g2.drawString(predText, tx, startY - 20);
            
            g2.setColor(Color.BLACK);
            int ax = drawX + (displaySize - fm.stringWidth(actText)) / 2;
            g2.drawString(actText, ax, startY + displaySize + 20);
        }
    }

    private void drawGraph(Graphics2D g2, int xOffset, int yOffset, int width, int height, 
                           List<Double> trainData, List<Double> testData, 
                           String title, String yLabelStr, 
                           double maxY, double tickStep, boolean isAcc) {
        
        int paddingLeft = 70; 
        int paddingRight = 40;
        int paddingTop = 60;
        int paddingBottom = 60;

        int graphX = xOffset + paddingLeft;
        int graphY = yOffset + paddingTop;
        int graphW = width - (paddingLeft + paddingRight);
        int graphH = height - (paddingTop + paddingBottom);

        int dataSize = trainData.size();
        int currentMax = dataSize - 1;
        if (currentMax < 0) currentMax = 0;
        
        int xAxisMax;
        if (this.maxEpochs > 0) {
            xAxisMax = this.maxEpochs; 
        } else {
            xAxisMax = (int) (Math.ceil(currentMax / 10.0) * 10);
            if (xAxisMax == 0) xAxisMax = 10;
        }

        g2.setColor(Color.BLACK);
        g2.setStroke(new BasicStroke(1.5f));
        g2.drawLine(graphX, graphY + graphH, graphX + graphW, graphY + graphH);
        g2.drawLine(graphX, graphY, graphX, graphY + graphH);

        g2.setFont(new Font("SansSerif", Font.BOLD, 18));
        FontMetrics fm = g2.getFontMetrics();
        int titleX = xOffset + (width - fm.stringWidth(title)) / 2;
        g2.drawString(title, titleX, yOffset + 35);

        g2.setFont(new Font("SansSerif", Font.BOLD, 14));
        fm = g2.getFontMetrics();
        AffineTransform originalTransform = g2.getTransform();
        g2.rotate(Math.toRadians(-90), graphX - 45, graphY + graphH / 2);
        g2.drawString(yLabelStr, graphX - 45 - fm.stringWidth(yLabelStr)/2, graphY + graphH / 2);
        g2.setTransform(originalTransform);

        String xLabel = "Epoch";
        fm = g2.getFontMetrics();
        g2.drawString(xLabel, graphX + (graphW - fm.stringWidth(xLabel)) / 2, graphY + graphH + 45);

        g2.setFont(new Font("SansSerif", Font.PLAIN, 12));
        int numTicks = (int)(maxY / tickStep);
        for (int i = 0; i <= numTicks; i++) {
            double val = i * tickStep;
            int py = graphY + graphH - (int)((val / maxY) * graphH);
            g2.drawLine(graphX - 5, py, graphX, py);
            String tickStr = String.format("%.1f", val);
            if (Math.abs(val) < 0.001) tickStr = "0.0";
            fm = g2.getFontMetrics();
            g2.drawString(tickStr, graphX - fm.stringWidth(tickStr) - 10, py + 5);
        }

        int xStep = 10; 
        for (int i = 0; i <= xAxisMax; i += xStep) {
            int px = graphX + (int)((double)i / xAxisMax * graphW);
            if (px > graphX + graphW) break; 
            
            g2.drawLine(px, graphY + graphH, px, graphY + graphH + 5);
            String tickStr = String.valueOf(i);
            fm = g2.getFontMetrics();
            g2.drawString(tickStr, px - fm.stringWidth(tickStr)/2, graphY + graphH + 20);
        }

        drawLinesAndPoints(g2, trainData, graphX, graphY, graphW, graphH, maxY, xAxisMax, Color.BLUE);
        drawLinesAndPoints(g2, testData, graphX, graphY, graphW, graphH, maxY, xAxisMax, Color.RED);

        int legendX = graphX + graphW - 110 - 10; 
        int legendY;
        if (isAcc) legendY = graphY + graphH - 50 - 10;
        else legendY = graphY + 10;
        drawLegend(g2, legendX, legendY);

        if (!trainData.isEmpty()) {
            double lastTrain = trainData.get(trainData.size() - 1);
            double lastTest = testData.get(testData.size() - 1);
            
            String metricName = isAcc ? "Accuracy" : "Loss";
            String trainText = String.format("Train %s : %.5f", metricName, lastTrain);
            String testText = String.format("Test %s : %.5f", metricName, lastTest);

            g2.setFont(new Font("SansSerif", Font.BOLD, 16));
            fm = g2.getFontMetrics();
            
            int cx = graphX + graphW / 2;
            int cy = graphY + graphH / 2;

            g2.setColor(Color.BLUE);
            g2.drawString(trainText, cx - fm.stringWidth(trainText) / 2, cy - 15);
            g2.setColor(Color.RED);
            g2.drawString(testText, cx - fm.stringWidth(testText) / 2, cy + 15);
        }
    }
    
    private void drawLinesAndPoints(Graphics2D g2, List<Double> data, int x, int y, int w, int h, double maxY, int xRange, Color c) {
        if (data.isEmpty()) return;
        g2.setColor(c);
        g2.setStroke(new BasicStroke(2f));
        
        int size = data.size();
        int[] xPoints = new int[size];
        int[] yPoints = new int[size];
        
        for (int i = 0; i < size; i++) {
            xPoints[i] = x + (int)((double)i / xRange * w);
            double val = Math.min(Math.max(data.get(i), 0), maxY);
            yPoints[i] = y + h - (int)((val / maxY) * h);
        }

        if (size > 1) {
            g2.drawPolyline(xPoints, yPoints, size);
        }
        for (int i = 0; i < size; i++) {
            g2.fillOval(xPoints[i] - 2, yPoints[i] - 2, 4, 4);
        }
    }

    private void drawLegend(Graphics2D g2, int x, int y) {
        g2.setColor(Color.LIGHT_GRAY);
        g2.drawRect(x - 10, y - 5, 110, 50);
        g2.setFont(new Font("SansSerif", Font.BOLD, 12));
        g2.setColor(Color.BLUE);
        g2.fillRect(x, y + 5, 10, 10);
        g2.setColor(Color.BLACK);
        g2.drawString("Train", x + 15, y + 15);
        g2.setColor(Color.RED);
        g2.fillRect(x, y + 25, 10, 10);
        g2.setColor(Color.BLACK);
        g2.drawString("Test", x + 15, y + 35);
    }
}