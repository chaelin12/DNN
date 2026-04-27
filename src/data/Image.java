package data;


public class Image
{
    private double[] data; 
    private int label;    

    
    public Image( double[] data, int label )
    {
        this.data = data;
         
        this.label = label;
    }
    
    public double[] getData() { return data; }// 그림 데이터

    public int getLabel() { return label; }//맞춰야 할 실제 숫자 값
    
    @Override
    public String toString() 
    {
        System.out.println( "Label: " + label );

        for( int i = 0; i < data.length; i++ )
        {
            if ( data.length % 28 == 0 )
                System.out.println();
            System.out.println( data[ i ] );
        }
        
        return "";
    }
}
