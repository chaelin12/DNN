package layers;

public class OutputLayer extends Layer{
    public OutputLayer(double[] input, int outputSize){
        super(input, outputSize);
    }

    public double calculateLoss(double[] target){
        //크로스 엔트로피 손실 함수 계산
        double loss = 0.0;
        for(int i=0; i<outputSize; i++){
            loss += - target[i] * Math.log(actOutputs[i] + 1e-15); //크로스 엔트로피 손실 함수
        }
        return loss;
    }


    public void calculateLocalGradients(double[] target){
        for(int i=0; i<outputSize; i++){
            dL_dz[i] = actOutputs[i] - target[i];
        }
        super.calculateLocalGradients();
    }

     @Override//Softmax Activation
        protected void activation( double[] preActOutput, double[] actOutput ) 
        {
            double max = Double.NEGATIVE_INFINITY;
            for ( int i = 0; i < outputSize; i++ )
                if ( preActOutput[ i ] > max )
                    max = preActOutput[ i ];
            double sum = 0.0; 
            for ( int i = 0; i < outputSize; i++ ) 
            {
                actOutput[ i ] = Math.exp( preActOutput[ i ] - max );
                sum += actOutput[ i ];
            }
            for ( int i = 0; i < outputSize; i++ )
                actOutput[ i ] /= sum;
        }
    
}
