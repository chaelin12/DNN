package layers;

public class HiddenLayer extends Layer{


    public HiddenLayer(double[] input, int outputSize){
        super(input, outputSize);
    }


    public void calculateLocalGradients(double[] dL_da){
        //은닉층의 델타 계산: ReLU dL/dz = dL/da * ReLU'
        for(int i=0; i<outputSize; i++){
            double gradient = preActOutputs[i] > 0 ? 1.0 : 0.01;
            this.dL_dz[i] = dL_da[i] * gradient;
        }
        super.calculateLocalGradients();
    }

    @Override
    protected void activation(double[] preActOutputs, double[] actOutputs){
        for(int i=0; i<preActOutputs.length; i++){
            actOutputs[i] = preActOutputs[i] > 0 ? preActOutputs[i] : 0.01 * preActOutputs[i];
        }
    }
    
}
