import {
  Image,
  NativeModules
} from 'react-native'
// Asumsi 'CommonFaceDetectionOptions' masih relevan untuk 'options'
import type {
  CommonFaceDetectionOptions
} from './FaceDetector' 

// --- Tipe Generik ---

type InputImage = number | string | { uri: string }

// --- Tipe untuk Pengenalan Wajah (recognizeFaces) ---

/**
 * Parameter untuk fungsi recognizeFaces.
 */
export interface ImageFaceRecognitionOptions {
  /** Gambar pertama untuk perbandingan. */
  image1: InputImage;
  /** Gambar kedua untuk perbandingan. */
  image2: InputImage;
  /** Opsi konfigurasi untuk face detector ML Kit. */
  options?: CommonFaceDetectionOptions;
}

/**
 * Objek hasil yang dikembalikan oleh recognizeFaces.
 * Sesuai dengan WritableMap dari kode native.
 */
export interface FaceRecognitionResult {
  /** Benar jika wajah dianggap cocok (berdasarkan DISTANCE_THRESHOLD). */
  match: boolean;
  /** * Persentase similaritas sebagai string (misal: "0.987").
   * Dihitung sebagai (1.0 - distance).
   */
  similarity: string;
  /** Jarak (distance) mentah antara dua embedding wajah (angka yang lebih kecil lebih mirip). */
  distance: number;
  /** Pesan error jika terjadi kegagalan selama proses. */
  error?: string | null;
}


// --- Fungsi Helper ---

/**
 * Menyelesaikan input gambar menjadi URI string.
 * * @param {InputImage} image Path gambar
 * @returns {string} URI yang telah diselesaikan
 */
function resolveUri( image: InputImage ): string {
  const uri = ( () => {
    switch ( typeof image ) {
      case 'number': {
        const source = Image.resolveAssetSource( image )
        return source?.uri
      }
      case 'string': {
        return image
      }
      case 'object': {
        return image?.uri
      }
      default: {
        return undefined
      }
    }
  } )()

  if ( !uri ) throw new Error( 'Unable to resolve image' )
  return uri
}

// --- Fungsi Native (recognizeFaces) ---

/**
 * Membandingkan dua gambar statis untuk mengenali apakah keduanya berisi wajah yang sama.
 * * @param {ImageFaceRecognitionOptions} params Parameter berisi image1, image2, dan options
 * @returns {Promise<FaceRecognitionResult>} Hasil perbandingan
 */
export async function recognizeFaces({
  image1,
  image2,
  options,
}: ImageFaceRecognitionOptions): Promise<FaceRecognitionResult> {
  // 1. Selesaikan kedua URI
  const uri1 = resolveUri(image1);
  const uri2 = resolveUri(image2);

  // 2. Dapatkan modul native
  // @ts-ignore
  const { ImageFaceDetector } = NativeModules;

  // 3. Periksa apakah fungsi 'recognizeFaces' ada
  if (!ImageFaceDetector?.recognizeFaces) {
    throw new Error(
      'Native module ImageFaceDetector.recognizeFaces() not found. Pastikan Anda telah me-rebuild aplikasi Anda.'
    );
  }

  // 4. Panggil fungsi native dengan kedua URI
  return await ImageFaceDetector.recognizeFaces(
    uri1,
    uri2,
    options
  );
}