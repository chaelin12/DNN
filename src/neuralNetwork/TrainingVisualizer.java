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
    
    // 마커 표시를 위한 변수
    private int bestEpoch = -1;
    private int overfitEpoch = -1;
    
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
    
    private List<ImageSample> currentSamples = new ArrayList<>();
    private JFrame frame;

    public TrainingVisualizer() {
    }

    // NeuralNetwork에서 호출하더라도 에러가 나지 않도록 껍데기만 유지 (가변 X축을 위해 사용 안 함)
    public void setMaxEpochs(int epochs) {
    }

    public void setMarkers(int bestEpoch, int overfitEpoch) {
        this.bestEpoch = bestEpoch;
        this.overfitEpoch = overfitEpoch;
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
            double maxVal = 0.0;
            for (double v : sample.pixels) if (v > maxVal) maxVal = v;
            if (maxVal <= 0) maxVal = 1.0; 

            BufferedImage bImg = new BufferedImage(imgSize, imgSize, BufferedImage.TYPE_INT_RGB);
            for (int r = 0; r < imgSize; r++) {
                for (int c = 0; c < imgSize; c++) {
                    double rawVal = sample.pixels[r * imgSize + c];
                    int gray = (int)((1.0 - (rawVal / maxVal)) * 255);
                    gray = Math.max(0, Math.min(255, gray));
                    int rgb = (0xFF << 24) | (gray << 16) | (gray << 8) | gray;
                    bImg.setRGB(c, r, rgb);
                }
            }
            g2.drawImage(bImg, drawX, startY, displaySize, displaySize, null);
            g2.setColor(Color.BLACK);
            g2.drawRect(drawX, startY, displaySize, displaySize); 

            String predText = "Pred: " + sample.predictedLabel;
            String actText = "Act: " + sample.actualLabel;
            g2.setFont(new Font("SansSerif", Font.BOLD, 14));
            g2.setColor(sample.predictedLabel == sample.actualLabel ? new Color(0, 150, 0) : Color.RED);
            FontMetrics fm = g2.getFontMetrics();
            g2.drawString(predText, drawX + (displaySize - fm.stringWidth(predText)) / 2, startY - 20);
            g2.setColor(Color.BLACK);
            g2.drawString(actText, drawX + (displaySize - fm.stringWidth(actText)) / 2, startY + displaySize + 20);
        }
    }

   private void drawGraph(Graphics2D g2, int xOffset, int yOffset, int width, int height, 
                           List<Double> trainData, List<Double> testData, 
                           String title, String yLabelStr, 
                           double maxY, double tickStep, boolean isAcc) {
        
        int paddingLeft = 80; 
        int paddingRight = 40; int paddingTop = 60; int paddingBottom = 60;
        int graphX = xOffset + paddingLeft;
        int graphY = yOffset + paddingTop;
        int graphW = width - (paddingLeft + paddingRight);
        int graphH = height - (paddingTop + paddingBottom);

        g2.setColor(Color.BLACK);
        g2.setFont(new Font("SansSerif", Font.BOLD, 18));
        FontMetrics fmTitle = g2.getFontMetrics();
        g2.drawString(title, xOffset + (width - fmTitle.stringWidth(title)) / 2, yOffset + 30);

        // ★ [수정됨] 데이터 개수에 따라 X축 가변 조절
        int dataCount = trainData.size();
        int xAxisMax = Math.max(10, dataCount - 1); // 최소 10칸 유지하면서, 데이터 늘어나면 유동적으로 팽창

        g2.setColor(Color.BLACK);
        g2.setStroke(new BasicStroke(1.5f));
        g2.drawLine(graphX, graphY + graphH, graphX + graphW, graphY + graphH); 
        g2.drawLine(graphX, graphY, graphX, graphY + graphH); 

        g2.setFont(new Font("SansSerif", Font.BOLD, 14));
        FontMetrics fmY = g2.getFontMetrics();
        AffineTransform oldAt = g2.getTransform();
        
        g2.rotate(-Math.PI / 2);
        int rotatedX = -(graphY + graphH / 2 + fmY.stringWidth(yLabelStr) / 2);
        int rotatedY = graphX - 55; 
        g2.drawString(yLabelStr, rotatedX, rotatedY);
        g2.setTransform(oldAt); 

        g2.setFont(new Font("SansSerif", Font.PLAIN, 12));
        for (double val = 0; val <= maxY + 0.0001; val += tickStep) {
            int py = graphY + graphH - (int)((val / maxY) * graphH);
            g2.drawLine(graphX - 5, py, graphX, py);
            g2.drawString(String.format("%.1f", val), graphX - 45, py + 5);
        }

        // X축 눈금 간격 설정 (데이터가 많아지면 눈금 간격을 넓게)
        int xTickStep = xAxisMax >= 20 ? 5 : 2; 
        for (int i = 0; i <= xAxisMax; i += xTickStep) {
            double ratio = (double) i / xAxisMax;
            int px = graphX + (int)(ratio * graphW);
            g2.drawLine(px, graphY + graphH, px, graphY + graphH + 5);
            String tickStr = String.valueOf(i);
            FontMetrics fm = g2.getFontMetrics();
            g2.drawString(tickStr, px - fm.stringWidth(tickStr)/2, graphY + graphH + 20);
        }
        g2.drawString("Epoch", graphX + graphW / 2 - 15, graphY + graphH + 45);

        drawLinesAndPoints(g2, trainData, graphX, graphY, graphW, graphH, maxY, xAxisMax, Color.BLUE);
        drawLinesAndPoints(g2, testData, graphX, graphY, graphW, graphH, maxY, xAxisMax, Color.RED);

        // 화살표 표시 (Test 데이터 기준)
        if (!isAcc && !testData.isEmpty()) {
            if (this.bestEpoch >= 0 && this.bestEpoch < testData.size()) {
                int pxBest = graphX + (int)((double)this.bestEpoch / xAxisMax * graphW);
                int pyBest = graphY + graphH - (int)(Math.min(testData.get(this.bestEpoch), maxY) / maxY * graphH);

                g2.setColor(new Color(0, 150, 0)); // 진한 초록색
                g2.drawLine(pxBest, pyBest - 50, pxBest, pyBest - 15);
                g2.fillPolygon(new int[]{pxBest - 5, pxBest + 5, pxBest}, new int[]{pyBest - 20, pyBest - 20, pyBest - 10}, 3);
                g2.setFont(new Font("SansSerif", Font.BOLD, 13));
                g2.drawString("⭐ Best", pxBest - 25, pyBest - 55);
            }

            if (this.overfitEpoch >= 0 && this.overfitEpoch < testData.size()) {
                int pxOver = graphX + (int)((double)this.overfitEpoch / xAxisMax * graphW);
                int pyOver = graphY + graphH - (int)(Math.min(testData.get(this.overfitEpoch), maxY) / maxY * graphH);

                g2.setColor(new Color(255, 100, 0)); // 주황색
                g2.drawLine(pxOver, pyOver + 50, pxOver, pyOver + 15);
                g2.fillPolygon(new int[]{pxOver - 5, pxOver + 5, pxOver}, new int[]{pyOver + 20, pyOver + 20, pyOver + 10}, 3);
                g2.setFont(new Font("SansSerif", Font.BOLD, 13));
                g2.drawString("⚠️ Overfit", pxOver - 30, pyOver + 65);
            }
        }

        if (!trainData.isEmpty() && !testData.isEmpty()) {
            g2.setFont(new Font("SansSerif", Font.BOLD, 15));
            String trVal = String.format("%s : %.5f", isAcc ? "Train Acc" : "Train Loss", trainData.get(trainData.size()-1));
            String teVal = String.format("%s : %.5f", isAcc ? "Test Acc" : "Test Loss", testData.get(testData.size()-1));
            
            g2.setColor(Color.BLUE);
            g2.drawString(trVal, graphX + graphW/2 - 50, graphY + graphH/2 - 20);
            g2.setColor(Color.RED);
            g2.drawString(teVal, graphX + graphW/2 - 50, graphY + graphH/2 + 10);
        }

        if (isAcc) {
            drawLegend(g2, graphX + graphW - 110, graphY + graphH - 55);
        } else {
            drawLegend(g2, graphX + graphW - 110, graphY + 10);
        }
    }
    
    private void drawLinesAndPoints(Graphics2D g2, List<Double> data, int x, int y, int w, int h, double maxY, int xRange, Color c) {
        if (data.isEmpty()) return;
        g2.setColor(c);
        int size = data.size();
        int[] xPoints = new int[size];
        int[] yPoints = new int[size];
        for (int i = 0; i < size; i++) {
            xPoints[i] = x + (int)((double)i / xRange * w);
            yPoints[i] = y + h - (int)(Math.min(data.get(i), maxY) / maxY * h);
        }
        if (size > 1) g2.drawPolyline(xPoints, yPoints, size);
        for (int i = 0; i < size; i++) g2.fillOval(xPoints[i]-3, yPoints[i]-3, 6, 6);
    }

    private void drawLegend(Graphics2D g2, int x, int y) {
        g2.setColor(new Color(255, 255, 255, 200));
        g2.fillRect(x, y, 100, 45);
        g2.setColor(Color.LIGHT_GRAY);
        g2.drawRect(x, y, 100, 45);
        g2.setFont(new Font("SansSerif", Font.BOLD, 12));
        g2.setColor(Color.BLUE); g2.fillRect(x + 5, y + 10, 10, 10);
        g2.setColor(Color.BLACK); g2.drawString("Train", x + 20, y + 20);
        g2.setColor(Color.RED); g2.fillRect(x + 5, y + 25, 10, 10);
        g2.setColor(Color.BLACK); g2.drawString("Test", x + 20, y + 35);
    }
}