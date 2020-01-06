import org.ejml.simple.SimpleMatrix;

import java.io.IOException;
import java.util.ArrayList;

public class NeuralNetwork {
    // init useful variables
    ArrayList<Layer> layers;
    String loss;
    SimpleMatrix output;
    MatrixMaths maths = new MatrixMaths();

    public NeuralNetwork(String lossFunction) {
        layers = new ArrayList<Layer>();
        loss = lossFunction;

    }

    public void addLayer(Layer layer) {
        layers.add(layer);
    }

    public ArrayList<SimpleMatrix> predict(ArrayList<SimpleMatrix> input) {
        int samples = input.size();
        int numLayers = layers.size();
        ArrayList<SimpleMatrix> result = new ArrayList<SimpleMatrix>();

        //forward propagate for sample and for each of the layers in the sample
        for (int i = 0; i < samples; i++){
            output = input.get(i);
            for (int j = 0; j < numLayers; j++) {
                output = layers.get(j).forwardPropagation(output);

            }
            result.add(output);
        }

        return result;
    }

    public int getNumLayers() {
        return layers.size();
    }

    public void fit(ArrayList<SimpleMatrix> xTrain, ArrayList<SimpleMatrix> yTrain, int epochs, double learningRate) {
        int samples = xTrain.size();
        double err = 0;
        SimpleMatrix backPropErr;
        int numLayers = layers.size();
    
        //forward propagate and back propagate through each sample for the required num epochs
        for (int i = 0; i < epochs; i++) {
            for (int j = 0; j < samples; j++) {
                output = xTrain.get(j);
                
                //forward propagate
                for (int k = 0; k < numLayers; k++) {
                    //System.out.print("Training input: ");
                    //maths.getDims(output);
                    output = layers.get(k).forwardPropagation(output);
                    //System.out.print("Training output: ");
                    //maths.getDims(output);
                }
                //System.out.println("***");

                err += mse(yTrain.get(j), output);
                //System.out.print("ytrain: ");
                //maths.getDims(yTrain.get(j));

                backPropErr = dmse(yTrain.get(j), output);
                for (int m = 0; m < numLayers; m++) {
                    //System.out.print("backprop out: ");
                    //maths.getDims(output);
                    //System.out.println((numLayers-m-1));
                    backPropErr = layers.get(numLayers - m - 1).backwardPropagation(backPropErr, learningRate);
                }
            }
            err = err/samples;
            System.out.println("epoch "+(i+1) + "/" + epochs + "  |  Error: " + err);
        }
    }

    private double mse(SimpleMatrix yTrue, SimpleMatrix yPred) {
        double mean = yTrue.minus(yPred).elementPower(2.0).elementSum();
        mean = mean / (double) yTrue.getNumElements();

        return mean;
    }

    private static SimpleMatrix dmse(SimpleMatrix yTrue, SimpleMatrix yPred) {
        int cols = yTrue.numCols();
        int rows = yTrue.numRows();
        int numElems = yTrue.getNumElements();
        double val;
        SimpleMatrix lossMatrix = new SimpleMatrix(new double[rows][cols]);

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                val = (yPred.get(i, j) - yTrue.get(i, j)) * 2.0/ (double) numElems;
                lossMatrix.set(i,j, val);
            }
        }

        return lossMatrix;
    }


    public static void main(String[] arguments) throws IOException
    {
//        double[][][] xData = {{{0,0}}, {{0,1}}, {{0,1}}, {{1,1}}};
//        double[][][] yData = {{{0}}, {{1}}, {{1}}, {{0}}};

//
//        for (int j = 0; j < xData.length; j++) {
//            xTrain.add(new SimpleMatrix(xData[j]));
//            yTrain.add(new SimpleMatrix(yData[j]));
//        }
        //Load data
        MNISTReader reader = new MNISTReader();
        reader.loadData();
        reader.extractData();
        reader.oneHotLabels();
        ArrayList<SimpleMatrix> xTrain = reader.getImages();
        ArrayList<SimpleMatrix> yTrain = reader.getLabels();
        
        //Add layers to network 
        NeuralNetwork network = new NeuralNetwork("mse");
        network.addLayer(new FullyConnectedLayer(28*28,256));
        network.addLayer(new ActivationLayer("tanh"));
        network.addLayer(new FullyConnectedLayer(256,10));
        network.addLayer(new ActivationLayer("tanh"));

        //System.out.println(network.getNumLayers());
        //train
        network.fit(xTrain, yTrain, 10, 0.01);
//
//        ArrayList<SimpleMatrix> out = network.predict(xTrain);
//
//        for (int k = 0; k <out.size(); k++) {
//            System.out.println(out.get(k));
//        }
    }


}
