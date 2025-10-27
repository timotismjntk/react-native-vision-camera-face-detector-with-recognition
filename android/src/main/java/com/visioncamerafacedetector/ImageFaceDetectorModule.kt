package com.visioncamerafacedetector

import android.util.Log
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Rect // <-- DITAMBAHKAN
import android.net.Uri
import com.facebook.react.bridge.*
import com.google.android.gms.tasks.Tasks // <-- DITAMBAHKAN
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.Face // <-- DITAMBAHKAN
import org.visioncamerafacedetector.recognizer.FaceRecognize // <-- DITAMBAHKAN
import java.io.InputStream
import java.net.HttpURLConnection
import java.net.URL
import java.lang.Exception // <-- DITAMBAHKAN
import kotlin.math.max // <-- DITAMBAHKAN

private const val TAG = "ImageFaceDetector"
class ImageFaceDetectorModule(
  private val reactContext: ReactApplicationContext
): ReactContextBaseJavaModule(reactContext) {
  override fun getName(): String = "ImageFaceDetector"

  // --- REFAKTOR: Inisialisasi helper secara lazy di level kelas ---
  /**
   * Inisialisasi FaceDetectorCommon.
   * Dibuat 'lazy' agar hanya diinisialisasi sekali saat pertama kali dibutuhkan.
   */
  private val faceDetectorCommon: FaceDetectorCommon by lazy {
    FaceDetectorCommon()
  }

  /**
   * Inisialisasi FaceRecognize.
   * Ini akan memanggil constructor dan memuat model TFLite.
   */
  private val faceRecognizer: FaceRecognize by lazy {
    Log.d(TAG, "Initializing FaceRecognize (loading TFLite model)...")
    try {
        FaceRecognize(reactContext.assets)
    } catch (e: Exception) {
        Log.e(TAG, "Failed to initialize FaceRecognize", e)
        throw e // Gagal jika model tidak bisa dimuat
    }
  }

  // --- REFAKTOR: Fungsi konversi data yang lebih bersih dan aman ---

  /**
   * Mengonversi ReadableMap rekursif menjadi Map<String, Any> Kotlin standar.
   */
  private fun convertReadableMapToMap(
    readableMap: ReadableMap?
  ): Map<String, Any> {
    val map = mutableMapOf<String, Any>()
    if (readableMap == null) return map

    val iterator = readableMap.keySetIterator()
    while (iterator.hasNextKey()) {
      val key = iterator.nextKey()
      when (readableMap.getType(key)) {
        ReadableType.Null -> map[key] = "" // Tetap seperti asli, atau bisa diganti null
        ReadableType.Boolean -> map[key] = readableMap.getBoolean(key)
        ReadableType.Number -> map[key] = readableMap.getDouble(key)
        ReadableType.String -> map[key] = readableMap.getString(key) ?: ""
        ReadableType.Map -> map[key] = convertReadableMapToMap(readableMap.getMap(key))
        ReadableType.Array -> map[key] = convertReadableArrayToList(readableMap.getArray(key))
      }
    }
    return map
  }

  /**
   * Mengonversi ReadableArray rekursif menjadi ArrayList<Any> Kotlin standar.
   */
  private fun convertReadableArrayToList(
    readableArray: ReadableArray?
  ): ArrayList<Any> {
    val list = arrayListOf<Any>()
    if (readableArray == null) return list

    for (i in 0 until readableArray.size()) {
      when (readableArray.getType(i)) {
        ReadableType.Null -> list.add("") // Tetap seperti asli, atau bisa diganti null
        ReadableType.Boolean -> list.add(readableArray.getBoolean(i))
        ReadableType.Number -> list.add(readableArray.getDouble(i))
        ReadableType.String -> list.add(readableArray.getString(i) ?: "")
        ReadableType.Map -> list.add(convertReadableMapToMap(readableArray.getMap(i)))
        ReadableType.Array -> list.add(convertReadableArrayToList(readableArray.getArray(i)))
      }
    }
    return list
  }

  /**
   * Mengonversi Map<String, Any> Kotlin standar menjadi WritableMap rekursif.
   */
  private fun toWritableMap(map: Map<String, Any>): WritableMap {
    val writableMap = Arguments.createMap()
    for ((key, value) in map) {
      @Suppress("UNCHECKED_CAST")
      when (value) {
        is Boolean -> writableMap.putBoolean(key, value)
        is Int -> writableMap.putInt(key, value)
        is Double -> writableMap.putDouble(key, value)
        is Float -> writableMap.putDouble(key, value.toDouble())
        is String -> writableMap.putString(key, value)
        is Map<*, *> -> writableMap.putMap(key, toWritableMap(value as Map<String, Any>))
        is List<*> -> writableMap.putArray(key, toWritableArray(value as List<Any?>))
        is ArrayList<*> -> writableMap.putArray(key, toWritableArray(value as List<Any?>))
        else -> writableMap.putNull(key)
      }
    }
    return writableMap
  }

  /**
   * Mengonversi List<Any?> Kotlin standar menjadi WritableArray rekursif.
   * Ini memperbaiki bug di kode asli yang hanya bisa menangani List<Map<...>>.
   */
  private fun toWritableArray(list: List<Any?>): WritableArray {
    val array = Arguments.createArray()
    for (value in list) {
      @Suppress("UNCHECKED_CAST")
      when (value) {
        is Boolean -> array.pushBoolean(value)
        is Int -> array.pushInt(value)
        is Double -> array.pushDouble(value)
        is Float -> array.pushDouble(value.toDouble())
        is String -> array.pushString(value)
        is Map<*, *> -> array.pushMap(toWritableMap(value as Map<String, Any>))
        is List<*> -> array.pushArray(toWritableArray(value as List<Any?>))
        is ArrayList<*> -> array.pushArray(toWritableArray(value as List<Any?>))
        else -> array.pushNull()
      }
    }
    return array
  }
  
  // --- REFAKTOR: `detectFaces` sekarang aman dari null dan me-recycle bitmap ---
  @ReactMethod
  fun detectFaces(
    uri: String, 
    options: ReadableMap?,
    promise: Promise
  ) {
    var bitmap: Bitmap? = null // Deklarasikan di sini untuk di-recycle
    try {
      // Gunakan helper yang sudah diinisialisasi
      val common = faceDetectorCommon 
      val (
        runContours,
        runClassifications,
        runLandmarks,
        trackingEnabled,
        faceDetector
      ) = common.getFaceDetector(
        convertReadableMapToMap(options) // Ganti nama fungsi
      )

      // REFAKTOR: Pemeriksaan null yang aman, hapus '!!'
      bitmap = loadBitmapFromUri(uri)
      if (bitmap == null) {
          promise.reject("IMAGE_LOAD_ERROR", "Gagal memuat gambar dari URI: $uri")
          return
      }

      val image = InputImage.fromBitmap(
        bitmap,
        0
      )
      faceDetector.process(image)
        .addOnSuccessListener { faces ->
          val result = common.processFaces(
            faces,
            runLandmarks,
            runClassifications,
            runContours,
            trackingEnabled
          )
          
          // Gunakan fungsi 'toWritableArray' yang sudah direfaktor
          // Asumsi 'result' adalah List<Map<String, Any>>
          promise.resolve(toWritableArray(result as List<Any?>))
        }
        .addOnFailureListener { e ->
          Log.e(TAG, "Error processing image face detection: ", e)
          promise.resolve(Arguments.createArray())
        }
        .addOnCompleteListener {
          // REFAKTOR: Pastikan bitmap di-recycle untuk mencegah kebocoran memori
          bitmap?.recycle()
        }
    } catch (e: Exception) {
      bitmap?.recycle() // Coba recycle jika terjadi error sinkron
      Log.e(TAG, "Error preparing face detection: ", e)
      promise.resolve(Arguments.createArray())
    }
  }

  // Fungsi ini sudah solid, hanya perlu penyesuaian kecil
  private fun cropBitmap(source: Bitmap, box: Rect): Bitmap {
    val left = max(0, box.left) // Gunakan max
    val top = max(0, box.top) // Gunakan max
    val width = if (left + box.width() > source.width) source.width - left else box.width()
    val height = if (top + box.height() > source.height) source.height - top else box.height()

    if (width <= 0 || height <= 0) {
        throw Exception("Invalid crop dimensions. Bounding box mungkin tidak valid.")
    }
    
    return Bitmap.createBitmap(source, left, top, width, height)
  }

  // --- REFAKTOR: `recognizeFaces` sekarang menginisialisasi model ---
  @ReactMethod
  fun recognizeFaces(
    uri: String,
    uri2: String,
    options: ReadableMap?,
    promise: Promise
) {
    try {
        // REFAKTOR: Panggil ini untuk MEMASTIKAN TFLite model dimuat
        // sebelum memanggil metode statis FaceRecognize.recognizeFace
        faceRecognizer

        // 1. Setup Face Detector dari ML Kit (gunakan helper lazy)
        val common = faceDetectorCommon 
        val (
            _, // runContours tidak dipakai
            _, // runClassifications tidak dipakai
            _, // runLandmarks tidak dipakai
            _, // trackingEnabled tidak dipakai
            faceDetector
        ) = common.getFaceDetector(
            convertReadableMapToMap(options) // Ganti nama fungsi
        )

        // 2. Load kedua bitmap
        val bitmap1 = loadBitmapFromUri(uri)
        val bitmap2 = loadBitmapFromUri(uri2)

        if (bitmap1 == null || bitmap2 == null) {
            promise.reject("IMAGE_LOAD_ERROR", "Gagal memuat satu atau kedua gambar dari URI.")
            return
        }

        // 3. Buat InputImage
        val image1 = InputImage.fromBitmap(bitmap1, 0)
        val image2 = InputImage.fromBitmap(bitmap2, 0)

        // 4. Proses kedua image secara PARALEL
        val task1 = faceDetector.process(image1)
        val task2 = faceDetector.process(image2)

        // 5. Gunakan Tasks.whenAllSuccess untuk menunggu kedua proses selesai
        Tasks.whenAllSuccess<List<Face>>(task1, task2)
            .addOnSuccessListener { results ->
                try {
                    val faces1 = results[0] // Hasil dari task1
                    val faces2 = results[1] // Hasil dari task2

                    // 6. Validasi HANYA satu wajah di setiap gambar
                    if (faces1.size != 1) {
                        promise.reject("FACE_COUNT_ERROR", "Gambar 1 harus memiliki tepat satu wajah. Ditemukan: ${faces1.size}.")
                        return@addOnSuccessListener
                    }
                    if (faces2.size != 1) {
                        promise.reject("FACE_COUNT_ERROR", "Gambar 2 harus memiliki tepat satu wajah. Ditemukan: ${faces2.size}.")
                        return@addOnSuccessListener
                    }

                    // 7. Dapatkan bounding box dan CROP bitmap
                    val face1 = faces1[0]
                    val face2 = faces2[0]

                    val croppedBitmap1 = cropBitmap(bitmap1, face1.boundingBox)
                    val croppedBitmap2 = cropBitmap(bitmap2, face2.boundingBox)

                    // 8. Panggil FaceRecognize.recognizeFace
                    val comparisonResult = FaceRecognize.recognizeFace(croppedBitmap1, croppedBitmap2)

                    // 9. Resolve promise dengan hasil perbandingan
                    promise.resolve(comparisonResult)

                } catch (e: Exception) {
                    // Tangani error cropping atau casting
                    promise.reject("PROCESS_ERROR", "Error saat cropping wajah: ${e.message}", e)
                } finally {
                    // 10. Bersihkan bitmap original setelah selesai
                    bitmap1.recycle()
                    bitmap2.recycle()
                    // (Bitmap yang di-crop akan di-recycle di dalam FaceRecognize.getFaceEmbedding)
                }
            }
            .addOnFailureListener { e ->
                // Tangani error dari ML Kit face detector
                Log.e(TAG, "Error processing image face detection: ", e)
                promise.reject("MLKIT_ERROR", "Deteksi wajah gagal untuk satu atau kedua gambar.", e)
            }
    } catch (e: Exception) {
        // Tangani error sinkron (loadBitmap, getFaceDetector, dll.)
        Log.e(TAG, "Error preparing face recognition: ", e)
        promise.reject("PREPARE_ERROR", "Error tidak terduga: ${e.message}", e)
    }
}

  // Fungsi ini sudah solid, tidak perlu diubah
  private fun loadBitmapFromUri(uriString: String): Bitmap? {
    return try {
      val uri = Uri.parse(uriString)
      when (uri.scheme?.lowercase()) {
        "content", "android.resource" -> {
          val stream = reactContext.contentResolver.openInputStream(uri)
          stream.useDecode()
        }
        "file" -> {
          val path = uri.path ?: return null
          if (path.startsWith("/android_asset/")) {
            val assetPath = path.removePrefix("/android_asset/")
            reactContext.assets.open(assetPath).useDecode()
          } else {
            BitmapFactory.decodeFile(path)
          }
        }
        "asset" -> {
          val assetPath = uriString.removePrefix("asset:/").removePrefix("asset:///")
          reactContext.assets.open(assetPath).useDecode()
        }
        "http", "https" -> {
          // Catatan: Operasi jaringan di thread utama bisa menyebabkan ANR.
          // React Native biasanya menangani ini, tapi perlu diperhatikan.
          val url = URL(uriString)
          val conn = url.openConnection() as HttpURLConnection
          conn.connect()
          val input = conn.inputStream
          input.useDecode()
        }
        else -> {
          // Fallback try as file path
          BitmapFactory.decodeFile(uriString)
        }
      }
    } catch (e: Exception) {
      Log.e(TAG, "Error loading bitmap from URI: $uriString", e)
      null
    }
  }
}

// Fungsi ini sudah solid, tidak perlu diubah
private fun InputStream?.useDecode(): Bitmap? {
  if (this == null) return null
  return try {
    this.use { 
      BitmapFactory.decodeStream(it)
    }
  } catch (e: Exception) {
    null
  }
}