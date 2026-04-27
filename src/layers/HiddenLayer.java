package layers;

public class HiddenLayer extends Layer{


    public HiddenLayer(double[] input, int outputSize){
        super(input, outputSize);
    }


    public void calculateLocalGradients(double[] dL_da){
        for(int i=0; i<outputSize; i++){
            double gradient = preActOutputs[i] > 0 ? 1.0 : 0;
            this.dL_dz[i] = dL_da[i] * gradient;
        }
        super.calculateLocalGradients();
    }

    @Override//ReLU
    protected void activation(double[] preActOutputs, double[] actOutputs){
        for(int i=0; i<preActOutputs.length; i++){
            actOutputs[i] = preActOutputs[i] > 0 ? preActOutputs[i] : 0 * preActOutputs[i];
        }
    }
    
}
