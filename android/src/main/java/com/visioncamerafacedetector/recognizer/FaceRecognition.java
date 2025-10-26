package org.visioncamerafacedetector.recognizer;

import android.content.SharedPreferences;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.net.Uri;
import android.os.Build;
import android.util.Base64;
import android.util.Log;
import android.util.Pair;

import com.facebook.react.bridge.Arguments;
import com.facebook.react.bridge.Promise;
import com.facebook.react.bridge.ReactApplicationContext;
import com.facebook.react.bridge.WritableArray;
import com.facebook.react.bridge.WritableMap;
import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.face.Face;
import com.google.mlkit.vision.face.FaceDetection;
import com.google.mlkit.vision.face.FaceDetector;
import com.google.mlkit.vision.face.FaceDetectorOptions;

import org.reactnative.facerecognizermodule.commons.TensorFlowLiteUtils;
import org.reactnative.facerecognizermodule.recognizer.utils.BitmapUtils;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.CompatibilityList;
import org.tensorflow.lite.gpu.GpuDelegate;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;

public class FaceRecognize {
    ReactApplicationContext mReactContext;
    private static Interpreter tfliteInterpreter;
    private static FaceDetector faceDetector;

    private static final String MODEL_FILE = "mobileFaceNet.tflite";
    private static final int OUTPUT_SIZE = 192;
    private static final int INPUT_SIZE = 112;
    private static final float IMAGE_MEAN = 128.0f;  // Confirm with your model
    private static final float IMAGE_STD = 128.0f;   // Confirm with your model
    private static final float DISTANCE_THRESHOLD = 0.9f;

    private static boolean isModelQuantized = false;  // Set true if your model is quantized

    private static int[] intValues = new int[INPUT_SIZE * INPUT_SIZE];
    private static float[][] embeddings;

    private static HashMap<String, SimilarityClassifier.Recognition> registeredFaceCollection = new HashMap<>();

    public FaceRecognize(AssetManager assetManager, ReactApplicationContext reactContext) throws IOException {
        mReactContext = reactContext;

        // Initialize FaceDetector once, reuse on each recognition or addFace call
        FaceDetectorOptions faceDetectorOptions = new FaceDetectorOptions.Builder()
                .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
                .setContourMode(FaceDetectorOptions.CONTOUR_MODE_NONE)
                .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_ALL)
                .setMinFaceSize(0.15f)
                .build();
        faceDetector = FaceDetection.getClient(faceDetectorOptions);

        // Init TensorFlow Lite Interpreter with GPU delegate if available, and 4 threads
        Interpreter.Options options = new Interpreter.Options();
        int numThreads = 4;  // Reduced from 64 for stability/performance balance

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
            options.setNumThreads(numThreads);
            tfliteInterpreter = new Interpreter(TensorFlowLiteUtils.loadModelFile(assetManager, MODEL_FILE), options);
        }
    }

    public WritableMap loadFaceCollectionFromSharedPref() {
        SharedPreferences sharedPreferences = mReactContext.getSharedPreferences("HashMap", mReactContext.MODE_PRIVATE);
        String defValue = new Gson().toJson(new HashMap<String, SimilarityClassifier.Recognition>());
        String json = sharedPreferences.getString("map", defValue);

        TypeToken<HashMap<String, SimilarityClassifier.Recognition>> token = new TypeToken<HashMap<String, SimilarityClassifier.Recognition>>() {};
        HashMap<String, SimilarityClassifier.Recognition> retrievedMap = new Gson().fromJson(json, token.getType());

        WritableMap response = Arguments.createMap();

        if (retrievedMap.isEmpty()) {
            response.putArray("registeredFace", Arguments.createArray());
        } else {
            WritableArray faceArray = Arguments.createArray();

            for (Map.Entry<String, SimilarityClassifier.Recognition> entry : retrievedMap.entrySet()) {
                float[][] output = new float[1][OUTPUT_SIZE];
                ArrayList arrayList = (ArrayList) entry.getValue().getExtra();
                arrayList = (ArrayList) arrayList.get(0);
                for (int counter = 0; counter < arrayList.size(); counter++) {
                    output[0][counter] = ((Double) arrayList.get(counter)).floatValue();
                }
                entry.getValue().setExtra(output);

                WritableMap recognitionMap = Arguments.createMap();
                SimilarityClassifier.Recognition recognition = entry.getValue();

                recognitionMap.putString("userId", recognition.userId);
                recognitionMap.putString("faceName", recognition.faceName);
                recognitionMap.putString("uri", recognition.uri);

                faceArray.pushMap(recognitionMap);
            }

            response.putArray("registeredFace", faceArray);
        }

        registeredFaceCollection = retrievedMap;

        return response;
    }

    public void saveFaceCollectionToSharedPref(int mode) {
        // mode: 0 = save all, 1 = clear all, 2 = update all
        if (mode == 1) {
            registeredFaceCollection.clear();
        } else if (mode == 0) {
            registeredFaceCollection.putAll(registeredFaceCollection);
        }
        String jsonString = new Gson().toJson(registeredFaceCollection);
        SharedPreferences sharedPreferences = mReactContext.getSharedPreferences("HashMap", mReactContext.MODE_PRIVATE);
        SharedPreferences.Editor editor = sharedPreferences.edit();
        editor.putString("map", jsonString);
        editor.apply();
    }

    private static Pair<String, Float> findNearestDistanceSimilarityFace(float[] emb) {
        Pair<String, Float> ret = null;
        for (Map.Entry<String, SimilarityClassifier.Recognition> entry : registeredFaceCollection.entrySet()) {
            final String name = entry.getKey();
            final float[] knownEmb = ((float[][]) entry.getValue().getExtra())[0];

            float distance = 0;
            for (int i = 0; i < emb.length; i++) {
                float diff = emb[i] - knownEmb[i];
                distance += diff * diff;
            }
            distance = (float) Math.sqrt(distance);

            if (ret == null || distance < ret.second) {
                ret = new Pair<>(name, distance);
            }
        }
        return ret;
    }

    public static WritableMap recognizeFace(Bitmap bitmap) {
        WritableMap response = Arguments.createMap();
        response.putString("similarity", "0");

        if (bitmap == null) {
            response.putString("detectedName", "invalid image");
            return response;
        }

        if (bitmap.getWidth() != INPUT_SIZE || bitmap.getHeight() != INPUT_SIZE) {
            bitmap = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, false);
        }

        bitmap.getPixels(intValues, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE);

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

        embeddings = new float[1][OUTPUT_SIZE];

        Object[] inputArray = {imgData};
        Map<Integer, Object> outputMap = new HashMap<>();
        outputMap.put(0, embeddings);

        tfliteInterpreter.runForMultipleInputsOutputs(inputArray, outputMap);

        Pair<String, Float> nearest = findNearestDistanceSimilarityFace(embeddings[0]);
        if (nearest != null) {
            if (nearest.second < DISTANCE_THRESHOLD) {
                response.putString("detectedName", nearest.first);
                response.putString("similarity", String.format(Locale.US, "%.3f", calculateSimilarityPercentage(nearest.second)));
            } else {
                response.putString("detectedName", "unknown");
                response.putString("similarity", String.format(Locale.US, "%.3f", calculateSimilarityPercentage(nearest.second)));
            }
        } else {
            response.putString("detectedName", "unknown");
        }

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

    public void addFace(String url, String userId, String faceName, Promise promise) throws IOException {
        WritableMap response = Arguments.createMap();
        try {
            Uri uri;
            BitmapUtils bitmapUtils = new BitmapUtils();

            if (registeredFaceCollection.containsKey(faceName)) {
                response.putString("addFace", "Error, face already registered with name " + faceName);
                promise.resolve(response);
                return;
            }

            // Check if the input is a URI string or a Base64-encoded image
            if (url.startsWith("data:image")) {
                // Input is a Base64-encoded image
                String base64Data = url.substring(url.indexOf(",") + 1);
                byte[] imageBytes = Base64.decode(base64Data, Base64.DEFAULT);

                String tempDir = mReactContext.getCacheDir().getPath();
                String imageType = url.split(";base64")[0].split("data:image/")[1];
                String tempFileName = "temp_image." + imageType; // You can choose the appropriate file format
                File tempFile = new File(tempDir, tempFileName);

                // Write the Base64 image data to a temporary file
                try (FileOutputStream outputStream = new FileOutputStream(tempFile)) {
                    outputStream.write(imageBytes);
                } catch (IOException e) {}

                // Create a Uri from the temporary file
                uri = Uri.fromFile(tempFile);
            } else if (url.startsWith("http")) {
                response.putString("addFace", "Error, URL not supported");
                promise.resolve(response);
                return;
            } else {
                uri = Uri.parse(url);
            }

            InputImage inputImage = InputImage.fromFilePath(mReactContext, uri);
            Bitmap bitmap = bitmapUtils.getBitmapFromUri(uri);

            FaceDetectorOptions highAccuracyOpts =
                    new FaceDetectorOptions.Builder()
                            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
                            .setContourMode(FaceDetectorOptions.CONTOUR_MODE_NONE)
                            .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_ALL)
                            .setMinFaceSize(0.15f)
                            .build();
            FaceDetector faceDetector = FaceDetection.getClient(highAccuracyOpts);

            faceDetector.process(inputImage).addOnSuccessListener(faces -> {
                if (faces.size() == 0) {
                    response.putString("addFace", "No face detected in picture");
                    promise.resolve(response);
                } else if (faces.size() == 1) {
                    Face face = faces.get(0);
                    //write code to recreate bitmap from source
                    //Write code to show bitmap to canvas

                    Bitmap frame_bmp1 = bitmapUtils.rotateBitmap(bitmap, 0, false, false);
                    RectF boundingBox = new RectF(face.getBoundingBox());
                    Bitmap cropped_face = bitmapUtils.getCropBitmapByCPU(frame_bmp1, boundingBox);
                    Bitmap scaled = bitmapUtils.getResizedBitmap(cropped_face, 112, 112);

                    recognizeFace(scaled);
                    //Create and Initialize new object with Face embeddings and Name.
                    Uri scaledUri = bitmapUtils.getUriFromBitmap(scaled, faceName);
                    SimilarityClassifier.Recognition result = new SimilarityClassifier.Recognition(
                            userId + "", faceName, -1f, scaledUri.toString());
                    result.setExtra(embeddings);
                    registeredFaceCollection.put(faceName, result);
                    response.putString("addFace", "Success add face");
                    bitmap.recycle();
                    frame_bmp1.recycle();
                    scaled.recycle();
                    promise.resolve(response);
                } else {
                    response.putString("addFace", "Error, make sure just one face in picture");
                    promise.resolve(response);
                }
            }).addOnFailureListener(e -> {
                response.putString("addFace", e.getMessage());
                promise.resolve(response);
            });
        } catch (IOException e) {
            // handle exception
            response.putString("addFace", e.getMessage());
            promise.resolve(response);
        }
    }


    public WritableMap deleteFace(String name) {
        WritableMap response = Arguments.createMap();

        if (registeredFaceCollection.containsKey(name)) {
            registeredFaceCollection.remove(name);
            // Save the updated registered faces to SharedPreferences
            saveFaceCollectionToSharedPref(2); // Use mode 2 to update the existing entries
            response.putBoolean("success", true);
            response.putString("message", "Face deleted successfully");
        } else {
            response.putBoolean("success", false);
            response.putString("message", "Face not found with the given name");
        }

        return response;
    }
}