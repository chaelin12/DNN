package neuralNetwork;
public class Main {
    public static void main(String[] args) {
        int BATCH =100;//배치 사이즈를 늘려 연산 효율 증가
        int EPOCHS = 40;//학습 횟수
        int[] hidden = {128, 64};//신경망 은닉층 노드 개수, 은닉층 2개
        NeuralNetwork nn = new NeuralNetwork(hidden, 0.003);//학습률 0.003
        nn.train_test(EPOCHS, BATCH);
        
    }
}
