import org.ejml.simple.SimpleMatrix;
import java.lang.Math;

public class ActivationLayer implements Layer {
    private String activation;
    private SimpleMatrix input;
    private SimpleMatrix output;
    MatrixMaths maths = new MatrixMaths();

    public ActivationLayer(String activationFunction) {
        activation = activationFunction;
    }

    public SimpleMatrix forwardPropagation(SimpleMatrix inputLayer) {
        //System.out.print("Input AL: ");
        //maths.getDims(inputLayer);

        input = inputLayer;
        if (activation == "tanh") {
            output = tanh(input);
        } else {
            System.out.println("Available activation functions: tanh");
        }

        return output;
    }

    public SimpleMatrix backwardPropagation(SimpleMatrix outputError, double learningRate) {
        //System.out.print("Input AL: ");
        //maths.getDims(outputError);
        SimpleMatrix activatedError = dTanh((input)).elementMult(outputError);
        //System.out.println(activatedError.get(0,0));
        //System.out.print("Output AL: ");
        //maths.getDims(activatedError);
        return activatedError;
    }

    private static SimpleMatrix tanh(SimpleMatrix matrix){
        int cols = matrix.numCols();
        int rows = matrix.numRows();
        SimpleMatrix activatedMatrix = new SimpleMatrix(new double[rows][cols]);

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                activatedMatrix.set(i,j, Math.tanh(matrix.get(i,j)));
            }
        }

        return activatedMatrix;
    }

    private static SimpleMatrix dTanh(SimpleMatrix matrix) {
        int cols = matrix.numCols();
        int rows = matrix.numRows();
        SimpleMatrix activatedMatrix = new SimpleMatrix(new double[rows][cols]);

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
//                System.out.println("*****");
//                System.out.println(matrix.get(i,j));
//                System.out.println((Math.pow(Math.tanh(matrix.get(i,j)),2)));
//                System.out.println(1 - (Math.pow(Math.tanh(matrix.get(i,j)),2)));
//                System.out.println("*****");
                activatedMatrix.set(i,j, 1 - Math.pow(Math.tanh(matrix.get(i,j)),2));
            }
        }

        return activatedMatrix;
    }

    public static void main(String[] arguments) {

        ActivationLayer a = new ActivationLayer("tanh");
        double[][] data = {{1.0, 2.0}};
        SimpleMatrix in = new SimpleMatrix(data);

        double[][] err = {{0.1, -0.2}};
        SimpleMatrix out = new SimpleMatrix(err);

        a.forwardPropagation(in);
        a.backwardPropagation(out, 0.4);

    }
}
