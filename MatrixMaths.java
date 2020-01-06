import org.ejml.simple.SimpleMatrix;
import java.util.Random;

public class MatrixMaths {


    public String getDims(SimpleMatrix matrix) {
        String values = Integer.toString(matrix.numRows()) + ", " +  Integer.toString(matrix.numCols());
        System.out.println(values);
        return values;
    }

    public double[][] initArray(double[][] array, double fillNumber){
        Random rand = new Random();

        int rows = array.length;
        int cols = array[0].length;

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                //to make sure they don't all become biased
                if (fillNumber == 0.0) {
                    array[i][j] = rand.nextDouble() - 0.5;
                } else {
                    array[i][j] = fillNumber;
                }
            }
        }
        return array;

    }
}
