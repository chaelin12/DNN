package layers;

public class HiddenLayer extends Layer{
    public HiddenLayer(double[] input, int outputSize){
        super(input, outputSize);
    }


    public void calculateLocalGradients(double[] dL_da){
        double[] da_dz = new double[outputSize];
        //은닉층의 델타 계산: ReLU dL/dz = dL/da * ReLU'
        for(int i=0; i<outputSize; i++){
            double reluDerivative = actOutputs[i] > 0 ? 1.0 : 0.0;
            da_dz[i] = dL_da[i] * reluDerivative;
        }

        Operations.computeDelta(dL_da, da_dz, this.dL_dz);

        super.calculateLocalGradients();
    }

    @Override
    protected void activation(double[] preActOutputs, double[] actOutputs){
        //ReLU 활성화 함수
        for(int i=0; i<preActOutputs.length; i++){
            actOutputs[i] = Math.max(0, preActOutputs[i]);// 둘 중에 더 큰 값 반환
        }
    } 
    
}
