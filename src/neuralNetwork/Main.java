package neuralNetwork;
public class Main {
    public static void main(String[] args) {
        int BATCH =10;
        int EPOCHS = 40;
        int[] hidden = {128, 64};
        NeuralNetwork nn = new NeuralNetwork(hidden, 0.001);
        nn.train_test(EPOCHS, BATCH);
    }
    
}
