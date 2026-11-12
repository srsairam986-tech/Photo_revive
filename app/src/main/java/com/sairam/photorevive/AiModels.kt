package com.sairam.photorevive

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import java.io.File
import java.nio.FloatBuffer
import kotlin.math.max
import kotlin.math.min

/** The free AI models ship inside the APK; they are copied once to app storage so ONNX Runtime can open them. */
object ModelFiles {
    fun path(ctx: Context, name: String): String {
        val f = File(ctx.filesDir, name)
        val size = try {
            ctx.assets.openFd("models/$name").use { it.length }
        } catch (e: Exception) {
            -1L
        }
        if (!f.exists() || (size > 0 && f.length() != size)) {
            val tmp = File(ctx.filesDir, "$name.tmp")
            ctx.assets.open("models/$name").use { input ->
                tmp.outputStream().use { out -> input.copyTo(out, 1 shl 20) }
            }
            tmp.renameTo(f)
        }
        return f.absolutePath
    }

    fun options(): OrtSession.SessionOptions = OrtSession.SessionOptions().apply {
        setIntraOpNumThreads(min(Runtime.getRuntime().availableProcessors(), 6))
        setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
    }
}

/** DDColor (tiny): predicts natural colours for a black & white photo. */
class Colorizer(ctx: Context) : AutoCloseable {
    private val env = OrtEnvironment.getEnvironment()
    private val session = env.createSession(ModelFiles.path(ctx, "colorize.onnx"), ModelFiles.options())

    /**
     * @param gray8 cleaned grayscale photo (any size)
     * @return the a and b colour channels at the same size as [gray8].
     * The AI works at 512 x 512; only its colour is used, all sharp detail comes from the original.
     */
    fun colorize(gray8: Mat): Pair<FloatArray, FloatArray> {
        val s = SIZE
        val small = Mat()
        Imgproc.resize(gray8, small, Size(s.toDouble(), s.toDouble()), 0.0, 0.0, Imgproc.INTER_AREA)
        val px = ByteArray(s * s)
        small.get(0, 0, px)
        small.release()
        val input = FloatArray(3 * s * s)
        for (i in 0 until s * s) {
            val v = (px[i].toInt() and 0xFF) / 255f
            input[i] = v; input[s * s + i] = v; input[2 * s * s + i] = v
        }
        val ab = OnnxTensor.createTensor(env, FloatBuffer.wrap(input), longArrayOf(1, 3, s.toLong(), s.toLong())).use { t ->
            session.run(mapOf("input" to t)).use { r ->
                val fb = (r[0] as OnnxTensor).floatBuffer
                FloatArray(fb.remaining()).also { fb.get(it) }
            }
        }
        val w = gray8.cols()
        val h = gray8.rows()
        fun channel(offset: Int): FloatArray {
            val m = Mat(s, s, CvType.CV_32F)
            m.put(0, 0, ab.copyOfRange(offset, offset + s * s))
            val o = Mat()
            Imgproc.resize(m, o, Size(w.toDouble(), h.toDouble()), 0.0, 0.0, Imgproc.INTER_CUBIC)
            val arr = FloatArray(w * h)
            o.get(0, 0, arr)
            m.release(); o.release()
            return arr
        }
        return Pair(channel(0), channel(s * s))
    }

    override fun close() = session.close()

    companion object { const val SIZE = 512 }
}

/** Real-ESRGAN general x4v3: 4x upscale that also removes blur, noise and JPEG blocks. */
class Upscaler(ctx: Context) : AutoCloseable {
    private val env = OrtEnvironment.getEnvironment()
    private val session = env.createSession(ModelFiles.path(ctx, "upscale4x.onnx"), ModelFiles.options())

    fun upscale(rgb8: Mat, onProgress: (Float) -> Unit): Bitmap {
        val w = rgb8.cols()
        val h = rgb8.rows()
        val src = ByteArray(w * h * 3)
        rgb8.get(0, 0, src)
        val out = Bitmap.createBitmap(w * 4, h * 4, Bitmap.Config.ARGB_8888)
        val tilesX = (w + TILE - 1) / TILE
        val tilesY = (h + TILE - 1) / TILE
        val total = tilesX * tilesY
        var done = 0
        for (ty in 0 until tilesY) for (tx in 0 until tilesX) {
            val x0 = tx * TILE
            val y0 = ty * TILE
            val x1 = min(x0 + TILE, w)
            val y1 = min(y0 + TILE, h)
            // overlap so tile borders don't show
            val px0 = max(x0 - PAD, 0)
            val py0 = max(y0 - PAD, 0)
            val px1 = min(x1 + PAD, w)
            val py1 = min(y1 + PAD, h)
            val pw = px1 - px0
            val ph = py1 - py0
            val plane = pw * ph
            val input = FloatArray(3 * plane)
            for (y in 0 until ph) {
                var s = ((py0 + y) * w + px0) * 3
                var d = y * pw
                for (x in 0 until pw) {
                    input[d] = (src[s].toInt() and 0xFF) / 255f
                    input[plane + d] = (src[s + 1].toInt() and 0xFF) / 255f
                    input[2 * plane + d] = (src[s + 2].toInt() and 0xFF) / 255f
                    s += 3; d++
                }
            }
            val res = OnnxTensor.createTensor(env, FloatBuffer.wrap(input), longArrayOf(1, 3, ph.toLong(), pw.toLong())).use { t ->
                session.run(mapOf("input" to t)).use { r ->
                    val fb = (r[0] as OnnxTensor).floatBuffer
                    FloatArray(fb.remaining()).also { fb.get(it) }
                }
            }
            val ow = pw * 4
            val oPlane = ow * ph * 4
            val cw = (x1 - x0) * 4
            val ch = (y1 - y0) * 4
            val offX = (x0 - px0) * 4
            val offY = (y0 - py0) * 4
            val pixels = IntArray(cw * ch)
            for (y in 0 until ch) {
                var i = (offY + y) * ow + offX
                var p = y * cw
                for (x in 0 until cw) {
                    val r = to8(res[i])
                    val g = to8(res[oPlane + i])
                    val b = to8(res[2 * oPlane + i])
                    pixels[p] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
                    i++; p++
                }
            }
            out.setPixels(pixels, 0, cw, x0 * 4, y0 * 4, cw, ch)
            done++
            onProgress(done.toFloat() / total)
        }
        return out
    }

    private fun to8(v: Float): Int = (v * 255f + 0.5f).toInt().coerceIn(0, 255)

    override fun close() = session.close()

    companion object {
        const val TILE = 192
        const val PAD = 10
    }
}
