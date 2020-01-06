import org.ejml.simple.SimpleMatrix;

public interface Layer {

    /**
     * Calculates output layer values from input layer
     */
    SimpleMatrix forwardPropagation(SimpleMatrix input);

    /**
     * Calculates error gradient during backward propagation
     * and updates values using gradient descent
     */
    SimpleMatrix backwardPropagation(SimpleMatrix outputError, double learningRate);
}