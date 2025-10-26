package org.visioncamerafacedetector.recognizer.utils;

import android.media.Image;

import java.nio.ByteBuffer;
import java.nio.ReadOnlyBufferException;

public class ImageUtils {
    static final int kMaxChannelValue = 262143;

    public void convertYUV420SPToARGB8888(byte[] input, int width, int height, int[] output) {
        final int frameSize = width * height;
        int yp = 0;

        for (int j = 0; j < height; j++) {
            int uvp = frameSize + (j >> 1) * width;
            int u = 0, v = 0;

            for (int i = 0; i < width; i++) {
                int y = input[yp] & 0xff;
                if ((i & 1) == 0) {
                    v = input[uvp++] & 0xff;
                    u = input[uvp++] & 0xff;
                }

                // Convert YUV to RGB inline:
                int y1192 = Math.max(0, y - 16) * 1192;
                int uShifted = u - 128;
                int vShifted = v - 128;

                int r = y1192 + 1634 * vShifted;
                int g = y1192 - 833 * vShifted - 400 * uShifted;
                int b = y1192 + 2066 * uShifted;

                // Clamp values to [0, kMaxChannelValue]
                r = r > kMaxChannelValue ? kMaxChannelValue : (r < 0 ? 0 : r);
                g = g > kMaxChannelValue ? kMaxChannelValue : (g < 0 ? 0 : g);
                b = b > kMaxChannelValue ? kMaxChannelValue : (b < 0 ? 0 : b);

                // Compose ARGB pixel
                output[yp++] = 0xff000000
                        | ((r >> 6) & 0xff0000)
                        | ((g >> 2) & 0xff00)
                        | ((b >> 10) & 0xff);
            }
        }
    }

    private static int YUV2RGB(int y, int u, int v) {
        // Adjust and check YUV values
        y = (y - 16) < 0 ? 0 : (y - 16);
        u -= 128;
        v -= 128;

        // This is the floating point equivalent. We do the conversion in integer
        // because some Android devices do not have floating point in hardware.
        // nR = (int)(1.164 * nY + 2.018 * nU);
        // nG = (int)(1.164 * nY - 0.813 * nV - 0.391 * nU);
        // nB = (int)(1.164 * nY + 1.596 * nV);
        int y1192 = 1192 * y;
        int r = (y1192 + 1634 * v);
        int g = (y1192 - 833 * v - 400 * u);
        int b = (y1192 + 2066 * u);

        // Clipping RGB values to be inside boundaries [ 0 , kMaxChannelValue ]
        r = r > kMaxChannelValue ? kMaxChannelValue : (r < 0 ? 0 : r);
        g = g > kMaxChannelValue ? kMaxChannelValue : (g < 0 ? 0 : g);
        b = b > kMaxChannelValue ? kMaxChannelValue : (b < 0 ? 0 : b);

        return 0xff000000 | ((r << 6) & 0xff0000) | ((g >> 2) & 0xff00) | ((b >> 10) & 0xff);
    }
}