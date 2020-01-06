import java.util.Random;
import org.ejml.simple.SimpleMatrix;

public class FullyConnectedLayer implements Layer {
    private int inputSize;
    private int outputSize;
    private SimpleMatrix weights;
    private SimpleMatrix bias;
    private SimpleMatrix input;
    private SimpleMatrix output;
    private MatrixMaths maths = new MatrixMaths();


    public FullyConnectedLayer(int inputLayerSize, int outputLayerSize) {
        //Define number of neurons in the input layer and the output layer
        inputSize = inputLayerSize;
        outputSize = outputLayerSize;

        //Init the weights and biases
        weights = new SimpleMatrix(maths.initArray(new double[inputSize][outputSize], 0.0));
        bias = new SimpleMatrix(maths.initArray(new double[1][outputSize], 0.0));
    }

    public SimpleMatrix forwardPropagation(SimpleMatrix inputLayer) {

        input = inputLayer;

        //System.out.print("Input FF: ");
        //maths.getDims(input);
        //System.out.print("Weights: ");
        //maths.getDims(weights);
        output = input.mult(weights).plus(bias);
        //System.out.print("Output FF: ");
        //maths.getDims(output);


        return output;
    }

    public SimpleMatrix backwardPropagation(SimpleMatrix outputError, double learningRate) {
        //System.out.print("OutputError: ");
        //maths.getDims(output);
        //System.out.print("Weights: ");
        //maths.getDims(weights);

        SimpleMatrix inputError = outputError.mult(weights.transpose());
        SimpleMatrix weightsError = input.transpose().mult(outputError);

        SimpleMatrix learningRateMat = new SimpleMatrix(maths.initArray(new double[weights.numRows()][weights.numCols()], learningRate));
        weightsError = weightsError.elementMult(learningRateMat);
        weights = weights.minus(weightsError);

        learningRateMat = new SimpleMatrix(maths.initArray(new double[bias.numRows()][bias.numCols()], learningRate));
        outputError = outputError.elementMult(learningRateMat);
        bias = bias.minus(outputError);

        return inputError;
    }

    public static void main(String[] arguments) {
        FullyConnectedLayer f = new FullyConnectedLayer(2,3);
        double[][] data = {{1.0, 2.0}};
        SimpleMatrix data2 = new SimpleMatrix(data);

        f.forwardPropagation(data2);
        double[][] err = {{0.1, -0.2, 1.1}};
        SimpleMatrix err2 = new SimpleMatrix(err);
        f.backwardPropagation(err2, 0.4);
    }
}
