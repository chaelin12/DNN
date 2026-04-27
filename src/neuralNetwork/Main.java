package neuralNetwork;
public class Main {
    public static void main(String[] args) {
        int BATCH = 128;//배치 사이즈
        int EPOCHS = 50;//학습횟수
        int[] hidden = {128, 64};//신경망 은닉층 노드 개수, 은닉층 2개
        NeuralNetwork nn = new NeuralNetwork(hidden, 0.001);//학습률 0.001
        nn.train_test(EPOCHS, BATCH);
    }
}
