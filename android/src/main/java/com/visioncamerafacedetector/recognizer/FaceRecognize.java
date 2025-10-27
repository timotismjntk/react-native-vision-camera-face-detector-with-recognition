package org.visioncamerafacedetector.recognizer;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.util.Log;

import com.facebook.react.bridge.Arguments;
import com.facebook.react.bridge.WritableMap;

import org.visioncamerafacedetector.recognizer.TensorFlowLiteUtils;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.CompatibilityList;
import org.tensorflow.lite.gpu.GpuDelegate;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;

public class FaceRecognize {
    private static Interpreter tfliteInterpreter;

    private static final String MODEL_FILE = "mobileFaceNet.tflite";
    private static final int OUTPUT_SIZE = 192;
    private static final int INPUT_SIZE = 112;
    private static final float IMAGE_MEAN = 128.0f;  // Confirm with your model
    private static final float IMAGE_STD = 128.0f;   // Confirm with your model
    private static final float DISTANCE_THRESHOLD = 0.9f; // Ambang batas untuk "match"

    private static boolean isModelQuantized = false;  // Set true if your model is quantized

    private static int[] intValues = new int[INPUT_SIZE * INPUT_SIZE];

    public FaceRecognize(AssetManager assetManager) throws IOException {
        // Init TensorFlow Lite Interpreter with GPU delegate if available, and 4 threads
        Interpreter.Options options = new Interpreter.Options();
        int numThreads = 4;

        try {
            CompatibilityList compatList = new CompatibilityList();

            if (compatList.isDelegateSupportedOnThisDevice()) {
                Log.i("rnfacerecognition", "GPU is supported, adding GPU delegate");
                GpuDelegate.Options delegateOptions = compatList.getBestOptionsForThisDevice();
                GpuDelegate gpuDelegate = new GpuDelegate(delegateOptions);
                options.addDelegate(gpuDelegate);
            } else {
                Log.i("rnfacerecognition", "GPU not supported, fallback to CPU/XNNPACK/NNAPI");
                options.setUseXNNPACK(true);
                options.setUseNNAPI(true);
            }
            options.setNumThreads(numThreads);

            tfliteInterpreter = new Interpreter(TensorFlowLiteUtils.loadModelFile(assetManager, MODEL_FILE), options);

        } catch (IOException e) {
            // Fallback ke CPU jika GPU init gagal
            options.setNumThreads(numThreads);
            tfliteInterpreter = new Interpreter(TensorFlowLiteUtils.loadModelFile(assetManager, MODEL_FILE), options);
        }
    }

    /**
     * Helper function to generate a face embedding from a bitmap.
     * @param bitmap The input bitmap (should be a cropped face).
     * @return A 2D float array (float[1][OUTPUT_SIZE]) representing the embedding, or null on failure.
     */
    private static float[][] getFaceEmbedding(Bitmap bitmap) {
        if (bitmap == null) {
            return null;
        }

        Bitmap scaledBitmap;
        if (bitmap.getWidth() != INPUT_SIZE || bitmap.getHeight() != INPUT_SIZE) {
            scaledBitmap = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, false);
        } else {
            scaledBitmap = bitmap;
        }

        // Gunakan intValues (static class member)
        scaledBitmap.getPixels(intValues, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE);

        ByteBuffer imgData = ByteBuffer.allocateDirect(INPUT_SIZE * INPUT_SIZE * 3 * 4);
        imgData.order(ByteOrder.nativeOrder());
        imgData.rewind();

        for (int i = 0; i < INPUT_SIZE; ++i) {
            for (int j = 0; j < INPUT_SIZE; ++j) {
                int pixelValue = intValues[i * INPUT_SIZE + j];
                if (isModelQuantized) {
                    // Quantized model
                    imgData.put((byte) ((pixelValue >> 16) & 0xFF));
                    imgData.put((byte) ((pixelValue >> 8) & 0xFF));
                    imgData.put((byte) (pixelValue & 0xFF));
                } else { // Float model normalization to roughly [-1,1]
                    imgData.putFloat((((pixelValue >> 16) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                    imgData.putFloat((((pixelValue >> 8) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                    imgData.putFloat(((pixelValue & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                }
            }
        }

        // Gunakan variabel lokal untuk output TFLite agar thread-safe
        float[][] localEmbeddings = new float[1][OUTPUT_SIZE];
        Object[] inputArray = {imgData};
        Map<Integer, Object> outputMap = new HashMap<>();
        outputMap.put(0, localEmbeddings);

        try {
            tfliteInterpreter.runForMultipleInputsOutputs(inputArray, outputMap);
        } catch (Exception e) {
            Log.e("FaceRecognize", "Error running TFLite interpreter: " + e.getMessage());
            return null;
        }
        
        // Recycle bitmap jika kita membuat salinan yang di-scaled
        if (scaledBitmap != bitmap) {
            scaledBitmap.recycle();
        }

        return localEmbeddings;
    }

    /**
     * Compares two bitmaps to check if they are the same face.
     * @param bitmap1 The first face bitmap.
     * @param bitmap2 The second face bitmap.
     * @return A WritableMap containing "match" (boolean), "similarity" (String %), and "distance" (double).
     */
    public static WritableMap recognizeFace(Bitmap bitmap1, Bitmap bitmap2) {
        WritableMap response = Arguments.createMap();

        if (bitmap1 == null || bitmap2 == null) {
            response.putBoolean("match", false);
            response.putString("similarity", "0.000");
            response.putDouble("distance", -1.0);
            response.putString("error", "Invalid input bitmap(s).");
            return response;
        }

        // 1. Dapatkan embedding untuk bitmap 1
        float[][] emb1_2d = getFaceEmbedding(bitmap1);
        if (emb1_2d == null) {
            response.putBoolean("match", false);
            response.putString("similarity", "0.000");
            response.putDouble("distance", -1.0);
            response.putString("error", "Failed to process bitmap 1.");
            return response;
        }
        float[] emb1 = emb1_2d[0];

        // 2. Dapatkan embedding untuk bitmap 2
        float[][] emb2_2d = getFaceEmbedding(bitmap2);
        if (emb2_2d == null) {
            response.putBoolean("match", false);
            response.putString("similarity", "0.000");
            response.putDouble("distance", -1.0);
            response.putString("error", "Failed to process bitmap 2.");
            return response;
        }
        float[] emb2 = emb2_2d[0];

        // 3. Hitung jarak (distance)
        float distance = 0;
        if (emb1.length != emb2.length) {
             response.putBoolean("match", false);
             response.putString("similarity", "0.000");
             response.putDouble("distance", -1.0);
             response.putString("error", "Embedding size mismatch.");
             return response;
        }

        for (int i = 0; i < emb1.length; i++) {
            float diff = emb1[i] - emb2[i];
            distance += diff * diff;
        }
        distance = (float) Math.sqrt(distance);

        // 4. Hitung persentase similaritas
        float similarityPercentage = calculateSimilarityPercentage(distance);

        // 5. Tentukan apakah match berdasarkan threshold
        boolean isMatch = distance < DISTANCE_THRESHOLD;

        response.putBoolean("match", isMatch);
        response.putString("similarity", String.format(Locale.US, "%.3f", similarityPercentage));
        response.putDouble("distance", (double) distance);
        response.putString("error", null);

        return response;
    }


    public static float calculateSimilarityPercentage(float v) {
        float similarityPercentage;
        if (v <= 1.0f) {
            similarityPercentage = 1.0f - v;
        } else {
            similarityPercentage = 0.0f; // Non-matching face
        }
        return similarityPercentage;
    }
}