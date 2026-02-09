package layers;

public class OutputLayer extends Layer{
    public OutputLayer(double[] input, int outputSize){
        super(input, outputSize);
    }

    public double computeLoss(double[] target){
        //크로스 엔트로피 손실 함수 계산
        double loss = 0.0;
        for(int i=0; i<outputSize; i++){
            loss += - target[i] * Math.log(actOutputs[i] + 1e-15); //크로스 엔트로피 손실 함수
        }
        return loss;
    }

    //델타 계산 오버로딩 -> 파라미터가 다르므로 재정의 X
    public void calculateLocalGradients(double[] target){
        //출력층의 델타 계산: Softmax dL/dz = a - y
        for(int i=0; i<outputSize; i++){
            dL_dz[i] = actOutputs[i] - target[i];
        }
        super.calculateLocalGradients();
    }

     @Override                                               // SOFTMAX ACTIVATION a = softmax(z) -> 1 x n
        protected void activation( double[] preActOutput, double[] actOutput ) 
        {
            double max = Double.NEGATIVE_INFINITY;
            
                                                        // Find maximum value for numerical stability
            for ( int i = 0; i < outputSize; i++ )
                
                if ( preActOutput[ i ] > max )
                    
                    max = preActOutput[ i ];
                
            
            double sum = 0.0;
            
                                                // Compute exponentials (shifted by max to avoid overflow)
            for ( int i = 0; i < outputSize; i++ ) 
            {
                actOutput[ i ] = Math.exp( preActOutput[ i ] - max );
                sum += actOutput[ i ];
            }
            
                                                // Normalize to probabilities
            for ( int i = 0; i < outputSize; i++ )
                
                actOutput[ i ] /= sum;
        }
    
}
