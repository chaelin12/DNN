package data;


import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;

public class DataReader
{


    private static final int dataLength = 784; // for mnist (28 x 28) GrayScale image
    

    public static List< Image > readData( String path ) 
    {
        List< Image > data = new ArrayList<>();    // list of images
        
        try ( BufferedReader reader = new BufferedReader( new FileReader( path ) ) )
        {
            String line;
            
            while ( ( line = reader.readLine() ) != null ) // each line is a 785 where 0 is label, rest is image
            {
                String[] values = line.split("," );
                
                if ( values.length < 2 ) continue;    // Skip empty lines
                
                int label = Integer.parseInt( values[ 0 ] );  // label is first element of line
                
                double[] pixels = new double[ dataLength ];      // to store image
                
                for ( int i = 1; i < values.length; i++ )
                    
                    pixels[ i-1 ] = Double.parseDouble( values[ i ]) / 255.0; // normalise [ 0 - 1 ]
                
        
                data.add( new Image( pixels, label ) ); // create new image with data, label and put in list
            }
        } 
        catch ( Exception e )
        {
            throw new IllegalArgumentException( "File not found " + path );
        }
        
        System.out.println( "Images loaded" );
        return data;  // return list of all images to caller
    }
}
