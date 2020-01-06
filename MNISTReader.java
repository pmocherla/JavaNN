import org.ejml.simple.SimpleMatrix;

import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.lang.reflect.Array;
import java.util.ArrayList;

public class MNISTReader {
    String dataPath;
    String labelPath;
    int numItems;
    ArrayList<SimpleMatrix> xTrain;
    ArrayList<SimpleMatrix> yTrain;
    ArrayList<Integer> labels;
    DataInputStream dataStream;
    DataInputStream labelDataStream;
    int nRows;
    int nCols;
    MatrixMaths maths = new MatrixMaths();


    public MNISTReader() {
        //http://yann.lecun.com/exdb/mnist/
        dataPath = "/Users/priyankamocherla/IdeaProjects/neuralNetwork/src/main/java/Data/t10k-images.idx3-ubyte";
        labelPath = "/Users/priyankamocherla/IdeaProjects/neuralNetwork/src/main/java/Data/t10k-labels.idx1-ubyte";
    }

    public void loadData() throws IOException {

        //Extract the info for training data
        FileInputStream file = new FileInputStream(dataPath);
        BufferedInputStream buffer = new BufferedInputStream(file);
        dataStream = new DataInputStream(buffer);

        int magicNumber = dataStream.readInt();
        int numberOfItems = dataStream.readInt();
        nRows = dataStream.readInt();
        nCols = dataStream.readInt();

        System.out.println("Magic number: " + magicNumber);
        System.out.println("Number of items: " + numberOfItems);
        System.out.println("Number of rows: " + nRows);
        System.out.println("Number of cols: " + nCols);

        numItems = numberOfItems;

        //label info
        FileInputStream labelFile = new FileInputStream(labelPath);
        BufferedInputStream labelBuffer = new BufferedInputStream(labelFile);
        labelDataStream = new DataInputStream(labelBuffer);

        magicNumber = labelDataStream.readInt();
        int labelNumberOfItems = labelDataStream.readInt();

        System.out.println("\nMagic number: " + magicNumber);
        System.out.println("Number of items: " + labelNumberOfItems);

        assert labelNumberOfItems == numberOfItems;
    }

    public void extractData() throws IOException {
        double[][] image = new double[1][nCols*nRows];
        labels = new ArrayList<Integer>();
        xTrain = new ArrayList<SimpleMatrix>();
        int items = nCols*nRows;

        for (int i = 0; i < numItems; i++) {
            labels.add(labelDataStream.readUnsignedByte());
            for (int j = 0; j < items; j++) {
                image[0][j] = dataStream.readUnsignedByte();
            }
            xTrain.add(new SimpleMatrix(image));
            //maths.getDims(xTrain.get(i));
        }

        dataStream.close();
        labelDataStream.close();

    }

    public void oneHotLabels() {
        double[][] oneHot = new double[1][10];
        yTrain = new ArrayList<SimpleMatrix>();

        for (int i = 0; i < numItems; i++) {
            for (int j = 0; j < 10; j++) {
                if (j == labels.get(i)) {
                    oneHot[0][j] = 1.0;
                }
                else {
                    oneHot[0][j] = 0.0;
                }
            }
            yTrain.add(new SimpleMatrix(oneHot));
        }
    }

    public ArrayList<SimpleMatrix> getLabels(){
        return yTrain;
    }

    public ArrayList<SimpleMatrix> getImages(){
        return xTrain;
    }

    public static void main(String[] arguments) throws IOException {
        MNISTReader reader = new MNISTReader();
        reader.loadData();
        reader.extractData();
        reader.oneHotLabels();
    }


}
