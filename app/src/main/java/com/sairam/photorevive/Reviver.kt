package com.sairam.photorevive

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.PointF
import android.media.FaceDetector
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

enum class Framing { FILL, FIT }

data class ReviveOptions(
    val framing: Framing = Framing.FILL,
    val eightK: Boolean = false,
    val strongClean: Boolean = false,
)

/**
 * Old photo -> clean, colourised, natural-light, AI-upscaled picture in 922:2048 shape.
 *   4K = 1728 x 3840,   8K = 3456 x 7680.   Everything runs offline on the phone.
 */
class Reviver(private val ctx: Context) {

    fun interface Progress { fun update(fraction: Float, message: String) }

    fun run(source: Bitmap, opt: ReviveOptions, progress: Progress, preview: (Bitmap) -> Unit): Bitmap {
        val tw = if (opt.eightK) 3456 else 1728
        val th = if (opt.eightK) 7680 else 3840

        progress.update(0.02f, "Preparing…")
        val rgba = Mat()
        Utils.bitmapToMat(source, rgba)
        val src = Mat()
        Imgproc.cvtColor(rgba, src, Imgproc.COLOR_RGBA2RGB)
        rgba.release()

        // 1. frame to 922:2048 and pick the AI working size (1/4 of the final size)
        val region: Mat
        val workW: Int
        val workH: Int
        if (opt.framing == Framing.FILL) {
            val box = ImageOps.cropBox(src.cols(), src.rows(), faceCenter(source))
            region = src.submat(box)
            workW = tw / 4; workH = th / 4
        } else {
            region = src
            val s = min(tw.toDouble() / src.cols(), th.toDouble() / src.rows())
            workW = max(1, (src.cols() * s / 4).roundToInt())
            workH = max(1, (src.rows() * s / 4).roundToInt())
        }
        val work = Mat()
        val shrinking = workW < region.cols()
        Imgproc.resize(region, work, Size(workW.toDouble(), workH.toDouble()), 0.0, 0.0,
            if (shrinking) Imgproc.INTER_AREA else Imgproc.INTER_LANCZOS4)
        if (region !== src) region.release()
        src.release()

        val original = ImageOps.rgbToLab(work)
        val mono = ImageOps.isMonochrome(original)

        // 2. remove scratches, dust, grain; soften shake
        progress.update(0.06f, "Removing scratches and noise…")
        val gray = Mat()
        Imgproc.cvtColor(work, gray, Imgproc.COLOR_RGB2GRAY)
        val clean = ImageOps.clean(gray, opt.strongClean)
        gray.release()
        val cleanRgb = Mat()
        Imgproc.cvtColor(clean, cleanRgb, Imgproc.COLOR_GRAY2RGB)
        val lab = ImageOps.rgbToLab(cleanRgb)
        cleanRgb.release()

        // 3. colour
        if (mono) {
            progress.update(0.15f, "Colourising with AI…")
            Colorizer(ctx).use { c ->
                val (a, b) = c.colorize(clean)
                a.copyInto(lab.a); b.copyInto(lab.b)
            }
        } else {
            // already a colour photo: keep its colours, only clean + relight + upscale
            original.a.copyInto(lab.a); original.b.copyInto(lab.b)
        }
        clean.release()

        // 4. natural window-light colour and brightness
        progress.update(0.35f, "Natural light colour…")
        ImageOps.naturalLight(lab)
        val colour = ImageOps.labToRgb(lab)
        work.release()
        preview(toBitmap(colour))

        // 5. AI upscale x4 (also sharpens blur and removes leftover grain)
        progress.update(0.40f, "AI upscaling to ${if (opt.eightK) "8K" else "4K"}…")
        val big = Upscaler(ctx).use { u ->
            u.upscale(colour) { f -> progress.update(0.40f + 0.58f * f, "AI upscaling… ${(f * 100).toInt()}%") }
        }

        // 6. final 922:2048 picture
        val result = if (opt.framing == Framing.FIT && (big.width != tw || big.height != th)) {
            progress.update(0.99f, "Finishing…")
            val bgMat = ImageOps.blurredBackground(colour, tw, th)
            val out = toBitmap(bgMat)
            bgMat.release()
            Canvas(out).drawBitmap(big, ((tw - big.width) / 2).toFloat(), ((th - big.height) / 2).toFloat(), null)
            big.recycle()
            out
        } else big
        colour.release()
        progress.update(1f, "Done")
        return result
    }

    private fun toBitmap(rgb: Mat): Bitmap {
        val bmp = Bitmap.createBitmap(rgb.cols(), rgb.rows(), Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(rgb, bmp)
        return bmp
    }

    /** Centre of the faces in the photo (Android's built-in offline face finder), or null. */
    private fun faceCenter(bmp: Bitmap): PointF? = try {
        val scale = min(1f, 480f / max(bmp.width, bmp.height))
        val w = max(2, (bmp.width * scale).toInt() and 1.inv())   // detector needs an even width
        val h = max(2, (bmp.height * scale).toInt())
        val small = Bitmap.createScaledBitmap(bmp, w, h, true).copy(Bitmap.Config.RGB_565, false)
        val faces = arrayOfNulls<FaceDetector.Face>(6)
        val n = FaceDetector(w, h, faces.size).findFaces(small, faces)
        small.recycle()
        if (n == 0) null else {
            var x = 0f
            var y = 0f
            val p = PointF()
            for (i in 0 until n) { faces[i]!!.getMidPoint(p); x += p.x; y += p.y }
            PointF(x / n / scale, y / n / scale)
        }
    } catch (e: Exception) {
        null
    }
}
