package data;


public class Image
{
    private double[] data; // vector data of length 784 
    private int label;     // true output label ( the correct numbers)

    
    public Image( double[] data, int label )
    {
        this.data = data;
         
        this.label = label;
    }
    
    public double[] getData() { return data; }

    public int getLabel() { return label; }
    
    @Override
    public String toString()  // print image in console
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
