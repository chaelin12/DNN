import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;

public class MnistLoader {

    // 파일 경로 설정 (프로젝트 루트 기준 data 폴더가 있어야 함)
    private static final String TRAIN_IMAGE_PATH = "./data/train-images.idx3-ubyte";
    private static final String TRAIN_LABEL_PATH = "./data/train-labels.idx1-ubyte";
    private static final String TEST_IMAGE_PATH = "./data/t10k-images.idx3-ubyte";
    private static final String TEST_LABEL_PATH = "./data/t10k-labels.idx1-ubyte";

    private static final int SIZE = 784; // 28x28
    private static final int NUM_TRAIN = 60000;
    private static final int NUM_TEST = 10000;

    // 데이터 저장용 배열
    public double[][] trainImages;
    public double[][] testImages;
    public int[] trainLabels;
    public int[] testLabels;

    public MnistLoader() {
        trainImages = new double[NUM_TRAIN][SIZE];
        testImages = new double[NUM_TEST][SIZE];
        trainLabels = new int[NUM_TRAIN];
        testLabels = new int[NUM_TEST];
    }

    public void loadMnist() {
        try {
            System.out.println("Loading Training Images...");
            readImages(TRAIN_IMAGE_PATH, trainImages);
            
            System.out.println("Loading Training Labels...");
            readLabels(TRAIN_LABEL_PATH, trainLabels);
            
            System.out.println("Loading Test Images...");
            readImages(TEST_IMAGE_PATH, testImages);
            
            System.out.println("Loading Test Labels...");
            readLabels(TEST_LABEL_PATH, testLabels);
            
            System.out.println("MNIST Data Loaded Successfully.");
        } catch (IOException e) {
            e.printStackTrace();
            System.err.println("Failed to load MNIST data. Check file paths.");
            System.exit(1);
        }
    }

    /**
     * 이미지 데이터 읽기 (idx3-ubyte)
     * Header: [Magic Number(4)] [Count(4)] [Rows(4)] [Cols(4)]
     */
    private void readImages(String filePath, double[][] buffer) throws IOException {
        try (DataInputStream dis = new DataInputStream(new FileInputStream(filePath))) {
            int magic = dis.readInt(); // Java는 Big-Endian이므로 바로 읽음
            if (magic != 2051) throw new IOException("Invalid Magic Number for Images: " + magic);

            int count = dis.readInt();
            int rows = dis.readInt();
            int cols = dis.readInt();

            System.out.printf("Reading %d images (%dx%d)...\n", count, rows, cols);

            for (int i = 0; i < count; i++) {
                for (int j = 0; j < rows * cols; j++) {
                    // Java의 byte는 signed(-128~127)이므로 0~255로 변환하기 위해 & 0xFF 사용
                    int unsignedByte = dis.readByte() & 0xFF;
                    buffer[i][j] = unsignedByte / 255.0; // 0.0 ~ 1.0 정규화
                }
            }
        }
    }

    /**
     * 라벨 데이터 읽기 (idx1-ubyte)
     * Header: [Magic Number(4)] [Count(4)]
     */
    private void readLabels(String filePath, int[] buffer) throws IOException {
        try (DataInputStream dis = new DataInputStream(new FileInputStream(filePath))) {
            int magic = dis.readInt();
            if (magic != 2049) throw new IOException("Invalid Magic Number for Labels: " + magic);

            int count = dis.readInt();
            System.out.printf("Reading %d labels...\n", count);

            for (int i = 0; i < count; i++) {
                buffer[i] = dis.readByte() & 0xFF;
            }
        }
    }

    /**
     * C코드의 print_mnist_pixel 구현
     */
    public void printMnistPixel(double[][] dataImage, int index) {
        System.out.printf("Image Index: %d\n", index);
        for (int j = 0; j < SIZE; j++) {
            System.out.printf("%1.1f ", dataImage[index][j]);
            if ((j + 1) % 28 == 0) System.out.println();
        }
        System.out.println();
    }

    /**
     * C코드의 print_mnist_label 구현
     */
    public void printLabel(int[] dataLabel, int index) {
        System.out.printf("Label[%d]: %d\n", index, dataLabel[index]);
    }

    /**
     * 이미지를 PGM 파일로 저장 (C코드의 save_mnist_pgm + save_image)
     */
    public void saveMnistPgm(double[][] dataImage, int index, String filename) {
        int width = 28;
        int height = 28;
        
        // 파일 이름 지정
        if (filename == null || filename.isEmpty()) {
            filename = "image_" + index + ".pgm";
        }

        try (BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream(filename))) {
            // PGM Header (P5: Binary PGM)
            String header = String.format("P5\n# Created by Java MNIST Loader\n%d %d\n255\n", width, height);
            bos.write(header.getBytes());

            for (int j = 0; j < width * height; j++) {
                int val = (int) (dataImage[index][j] * 255.0);
                bos.write(val);
            }
            System.out.println("Image saved to " + filename);
        } catch (IOException e) {
            System.err.println("Could not save image.");
            e.printStackTrace();
        }
    }

    // 메인 실행 함수
    public static void main(String[] args) {
        MnistLoader loader = new MnistLoader();
        loader.loadMnist();

        // 테스트: 첫 번째 훈련 이미지의 픽셀값과 라벨 출력
        loader.printLabel(loader.trainLabels, 0);
        // loader.printMnistPixel(loader.trainImages, 0); // 콘솔이 꽉 차므로 필요시 주석 해제

        // 테스트: 이미지 파일로 저장 (.pgm)
        loader.saveMnistPgm(loader.trainImages, 0, "train_0.pgm");
    }
}