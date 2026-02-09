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
import java.util.ArrayList;
import java.util.List;
import javax.swing.JFrame;
import javax.swing.JPanel;

public class TrainingVisualizer extends JPanel {
    private List<Double> trainLoss = new ArrayList<>();
    private List<Double> testLoss = new ArrayList<>();
    private List<Double> trainAcc = new ArrayList<>();
    private List<Double> testAcc = new ArrayList<>();
    
    private int maxEpochs = 0; // 총 에포크 수 (X축 고정용)
    private JFrame frame;

    public TrainingVisualizer() {
        trainLoss.add(0.0); testLoss.add(0.0);
        trainAcc.add(0.0); testAcc.add(0.0);
    }

    // NeuralNetwork 생성자에서 호출 (X축 고정)
    public void setMaxEpochs(int epochs) {
        this.maxEpochs = epochs;
        repaint();
    }

    public void showWindow() {
        frame = new JFrame("Training Monitor");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        
        int windowWidth = 1200;
        int windowHeight = 600;
        frame.setSize(windowWidth, windowHeight);
        frame.add(this);

        // 창을 화면 우측 하단에 배치
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

    public void updateData(List<Double> trL, List<Double> teL, List<Double> trA, List<Double> teA) {
        this.trainLoss = new ArrayList<>(trL);
        this.testLoss = new ArrayList<>(teL);
        this.trainAcc = new ArrayList<>(trA);
        this.testAcc = new ArrayList<>(teA);
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
        int graphWidth = w / 2;
        
        // 1. Loss 그래프 (왼쪽) -> isAcc = false
        drawGraph(g2, 0, 0, graphWidth, h, 
                  trainLoss, testLoss, 
                  "Train & Test Loss", "Loss", 
                  3.0, 0.6, false);

        // 2. Accuracy 그래프 (오른쪽) -> isAcc = true
        drawGraph(g2, graphWidth, 0, graphWidth, h, 
                  trainAcc, testAcc, 
                  "Train & Test Accuracy", "Accuracy", 
                  1.0, 0.2, true);
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

        // X축 범위 계산
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

        // --- 1. 축 그리기 ---
        g2.setColor(Color.BLACK);
        g2.setStroke(new BasicStroke(1.5f));
        g2.drawLine(graphX, graphY + graphH, graphX + graphW, graphY + graphH);
        g2.drawLine(graphX, graphY, graphX, graphY + graphH);

        // --- 2. 그래프 제목 ---
        g2.setFont(new Font("SansSerif", Font.BOLD, 18));
        FontMetrics fm = g2.getFontMetrics();
        int titleX = xOffset + (width - fm.stringWidth(title)) / 2;
        g2.drawString(title, titleX, yOffset + 35);

        // --- 3. Y축 제목 ---
        g2.setFont(new Font("SansSerif", Font.BOLD, 14));
        fm = g2.getFontMetrics();
        AffineTransform originalTransform = g2.getTransform();
        g2.rotate(Math.toRadians(-90), graphX - 45, graphY + graphH / 2);
        g2.drawString(yLabelStr, graphX - 45 - fm.stringWidth(yLabelStr)/2, graphY + graphH / 2);
        g2.setTransform(originalTransform);

        // --- 4. X축 제목 ---
        String xLabel = "Epoch";
        fm = g2.getFontMetrics();
        g2.drawString(xLabel, graphX + (graphW - fm.stringWidth(xLabel)) / 2, graphY + graphH + 45);

        // --- 5. Y축 눈금 ---
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

        // --- 6. X축 눈금 ---
        int xStep = 10; 
        for (int i = 0; i <= xAxisMax; i += xStep) {
            int px = graphX + (int)((double)i / xAxisMax * graphW);
            if (px > graphX + graphW) break; 
            
            g2.drawLine(px, graphY + graphH, px, graphY + graphH + 5);
            String tickStr = String.valueOf(i);
            fm = g2.getFontMetrics();
            g2.drawString(tickStr, px - fm.stringWidth(tickStr)/2, graphY + graphH + 20);
        }

        // --- 7. 데이터 선 ---
        drawLinesAndPoints(g2, trainData, graphX, graphY, graphW, graphH, maxY, xAxisMax, Color.BLUE);
        drawLinesAndPoints(g2, testData, graphX, graphY, graphW, graphH, maxY, xAxisMax, Color.RED);

        // --- 8. [수정됨] 범례 (Legend) 위치 스마트 배치 ---
        // 박스 크기: 110 x 50, 여백 10px
        int legendX = graphX + graphW - 110 - 10; // 오른쪽 정렬
        int legendY;

        if (isAcc) {
            // Accuracy 그래프: 값이 위로 올라가므로 범례는 '아래'에 위치
            legendY = graphY + graphH - 50 - 10;
        } else {
            // Loss 그래프: 값이 아래로 떨어지므로 범례는 '위'에 위치 (요청하신 부분)
            legendY = graphY + 10;
        }
        drawLegend(g2, legendX, legendY);

        // --- 9. 중앙 텍스트 ---
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