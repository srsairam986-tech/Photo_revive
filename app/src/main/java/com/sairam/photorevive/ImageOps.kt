package com.sairam.photorevive

import android.graphics.PointF
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Rect
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import org.opencv.photo.Photo
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.hypot
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.roundToInt
import kotlin.math.sqrt
import kotlin.math.tanh

/** Photo in Lab colour space, one float array per channel (L 0..100, a/b about -128..127). */
class LabImage(val w: Int, val h: Int, val L: FloatArray, val a: FloatArray, val b: FloatArray)

/** Classic (non-AI) image steps: scratch removal, de-noise, natural-light colour, framing. */
object ImageOps {

    // Look of the reference photo (sunlit window, warm skin, bright clean whites)
    private const val REF_LO = 4f
    private const val REF_HI = 94f
    private const val REF_MED = 53.5f
    private const val REF_CHROMA = 13f
    private const val REF_HI_A = 0.5f
    private const val REF_HI_B = 3f

    fun release(vararg mats: Mat) = mats.forEach { it.release() }

    // ---------------------------------------------------------------- cleanup

    /** How grainy the photo is (Immerkaer's method). */
    fun noiseSigma(gray: Mat): Double {
        val k = Mat(3, 3, CvType.CV_32F)
        k.put(0, 0, floatArrayOf(1f, -2f, 1f, -2f, 4f, -2f, 1f, -2f, 1f))
        val r = Mat()
        Imgproc.filter2D(gray, r, CvType.CV_32F, k)
        val w = gray.cols()
        val h = gray.rows()
        val inner = r.submat(1, h - 1, 1, w - 1)
        val abs = Mat()
        Core.absdiff(inner, Scalar(0.0), abs)
        val sum = Core.sumElems(abs).`val`[0]
        release(k, inner, r, abs)
        return sqrt(PI / 2) * sum / (6.0 * (w - 2) * (h - 2))
    }

    /** White/black scratch lines (and dust specks in strong mode) as a 0/255 mask. */
    fun scratchMask(gray: Mat, sigma: Double, strong: Boolean): Mat {
        val rows = gray.rows()
        val cols = gray.cols()
        val n = rows * cols
        val sm = Mat()
        Imgproc.GaussianBlur(gray, sm, Size(0.0, 0.0), 0.8 + 0.08 * sigma)
        val med = Mat()
        Imgproc.medianBlur(sm, med, 9)            // removes lines up to ~4 px wide
        val s = ByteArray(n)
        val m = ByteArray(n)
        sm.get(0, 0, s)
        med.get(0, 0, m)
        release(sm, med)

        val thr = max(16.0, 1.6 * sigma)
        val bright = ByteArray(n)
        val dark = ByteArray(n)
        for (i in 0 until n) {
            val d = (s[i].toInt() and 0xFF) - (m[i].toInt() and 0xFF)
            if (d > thr) bright[i] = 1 else if (-d > thr) dark[i] = 1
        }

        val mask = ByteArray(n)
        val labelArr = IntArray(n)
        for (cand in arrayOf(bright, dark)) {
            val c = Mat(rows, cols, CvType.CV_8U)
            c.put(0, 0, cand)
            val labels = Mat()
            val stats = Mat()
            val cents = Mat()
            val count = Imgproc.connectedComponentsWithStats(c, labels, stats, cents, 8, CvType.CV_32S)
            val keep = BooleanArray(count)
            for (i in 1 until count) {
                val w = stats.get(i, Imgproc.CC_STAT_WIDTH)[0]
                val h = stats.get(i, Imgproc.CC_STAT_HEIGHT)[0]
                val area = stats.get(i, Imgproc.CC_STAT_AREA)[0]
                val length = hypot(w, h)
                val thick = area / max(length, 1.0)
                val fill = area / (w * h)
                if (length >= 30 && thick <= 4.5 && (fill < 0.35 || min(w, h) <= 5)) {
                    keep[i] = true                                   // long thin line = scratch
                } else if (strong && area in 2.0..30.0 && max(w, h) <= 7) {
                    keep[i] = true                                   // dust speck
                }
            }
            labels.get(0, 0, labelArr)
            for (i in 0 until n) if (keep[labelArr[i]]) mask[i] = 255.toByte()
            release(c, labels, stats, cents)
        }
        val out = Mat(rows, cols, CvType.CV_8U)
        out.put(0, 0, mask)
        val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(5.0, 5.0))
        Imgproc.dilate(out, out, kernel)
        kernel.release()
        return out
    }

    /**
     * Runs on the already de-noised photo, where thin scratches stand out from the grain.
     * Only long, straight, thin lines count (real details like lips or necklaces are kept);
     * strong mode also removes isolated dust specks.
     */
    private fun straightScratchAndDustMask(den: Mat, strong: Boolean): Mat {
        val rows = den.rows()
        val cols = den.cols()
        val n = rows * cols
        val med = Mat()
        Imgproc.medianBlur(den, med, 7)
        val s = ByteArray(n)
        val m = ByteArray(n)
        den.get(0, 0, s)
        med.get(0, 0, m)
        med.release()
        val d = IntArray(n) { (s[it].toInt() and 0xFF) - (m[it].toInt() and 0xFF) }

        val mask = ByteArray(n)
        val labelArr = IntArray(n)
        for (sign in intArrayOf(1, -1)) {
            val cand = ByteArray(n)
            for (i in 0 until n) if (sign * d[i] > 20) cand[i] = 1
            val c = Mat(rows, cols, CvType.CV_8U)
            c.put(0, 0, cand)
            val labels = Mat()
            val stats = Mat()
            val cents = Mat()
            val count = Imgproc.connectedComponentsWithStats(c, labels, stats, cents, 8, CvType.CV_32S)
            val keep = BooleanArray(count)
            for (i in 1 until count) {
                val x = stats.get(i, Imgproc.CC_STAT_LEFT)[0].toInt()
                val y = stats.get(i, Imgproc.CC_STAT_TOP)[0].toInt()
                val w = stats.get(i, Imgproc.CC_STAT_WIDTH)[0]
                val h = stats.get(i, Imgproc.CC_STAT_HEIGHT)[0]
                val area = stats.get(i, Imgproc.CC_STAT_AREA)[0]
                val length = hypot(w, h)
                if (length >= 60 && max(w, h) >= 6 * max(min(w, h), 1.0) && area / length <= 3) {
                    keep[i] = true
                } else if (strong && area <= 20 && max(w, h) <= 6) {
                    var peak = 0
                    for (yy in y until y + h.toInt()) for (xx in x until x + w.toInt()) {
                        peak = max(peak, abs(d[yy * cols + xx]))
                    }
                    if (peak > 45) keep[i] = true
                }
            }
            labels.get(0, 0, labelArr)
            for (i in 0 until n) if (keep[labelArr[i]]) mask[i] = 255.toByte()
            release(c, labels, stats, cents)
        }
        val out = Mat(rows, cols, CvType.CV_8U)
        out.put(0, 0, mask)
        val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(5.0, 5.0))
        Imgproc.dilate(out, out, kernel)
        kernel.release()
        return out
    }

    /** Remove scratches, film grain and soften camera shake on a grayscale photo. */
    fun clean(gray: Mat, strong: Boolean): Mat {
        val sigma = noiseSigma(gray)
        var cur = gray.clone()
        val mask = scratchMask(cur, sigma, strong)
        if (Core.countNonZero(mask) > 0) {
            val fixed = Mat()
            Photo.inpaint(cur, mask, fixed, 4.0, Photo.INPAINT_TELEA)
            cur.release()
            cur = fixed
        }
        mask.release()
        val h = (1.15 * sigma).coerceIn(3.0, 16.0).toFloat()
        var dn = Mat()
        Photo.fastNlMeansDenoising(cur, dn, h, 7, 15)
        // second pass on the de-noised photo: thin straight scratches (+ dust in strong mode)
        val mask2 = straightScratchAndDustMask(dn, strong)
        if (Core.countNonZero(mask2) > 0) {
            val fixed = Mat()
            Photo.inpaint(dn, mask2, fixed, 4.0, Photo.INPAINT_TELEA)
            dn.release()
            dn = fixed
        }
        mask2.release()
        // gentle sharpening for shake; the AI upscaler recovers the rest
        val blur = Mat()
        Imgproc.GaussianBlur(dn, blur, Size(0.0, 0.0), 1.6)
        val out = Mat()
        Core.addWeighted(dn, 1.45, blur, -0.45, 0.0, out)
        release(cur, dn, blur)
        return out
    }

    // ---------------------------------------------------------------- colour

    fun rgbToLab(rgb8: Mat): LabImage {
        val f = Mat()
        rgb8.convertTo(f, CvType.CV_32FC3, 1.0 / 255)
        val lab = Mat()
        Imgproc.cvtColor(f, lab, Imgproc.COLOR_RGB2Lab)
        val n = rgb8.rows() * rgb8.cols()
        val all = FloatArray(n * 3)
        lab.get(0, 0, all)
        release(f, lab)
        val l = FloatArray(n)
        val a = FloatArray(n)
        val b = FloatArray(n)
        for (i in 0 until n) {
            l[i] = all[3 * i]; a[i] = all[3 * i + 1]; b[i] = all[3 * i + 2]
        }
        return LabImage(rgb8.cols(), rgb8.rows(), l, a, b)
    }

    fun labToRgb(img: LabImage): Mat {
        val n = img.w * img.h
        val all = FloatArray(n * 3)
        for (i in 0 until n) {
            all[3 * i] = img.L[i].coerceIn(0f, 100f)
            all[3 * i + 1] = img.a[i]
            all[3 * i + 2] = img.b[i]
        }
        val m = Mat(img.h, img.w, CvType.CV_32FC3)
        m.put(0, 0, all)
        val rgb = Mat()
        Imgproc.cvtColor(m, rgb, Imgproc.COLOR_Lab2RGB)
        val out = Mat()
        rgb.convertTo(out, CvType.CV_8UC3, 255.0)
        release(m, rgb)
        return out
    }

    /** True for black & white and sepia photos. */
    fun isMonochrome(img: LabImage): Boolean {
        val c = sample(img.L.size) { i -> hypot(img.a[i], img.b[i]) }
        c.sort()
        return pct(c, 0.9) < 12f && c.average() < 8.0
    }

    /** Natural window-light look like the reference: normal colour balance, bright clean whites, soft contrast. */
    fun naturalLight(img: LabImage, strength: Float = 1f) {
        val l = img.L
        val a = img.a
        val b = img.b
        val n = l.size

        // 0. remove overall colour cast (orange / pink / blue) -> normal colours like the reference
        whiteBalance(img, 0.9f * strength)

        // 1. brightness & contrast towards the reference
        val sl = sample(n) { l[it] }
        sl.sort()
        val lo = pct(sl, 0.01)
        val hi = pct(sl, 0.99)
        val span = max(hi - lo, 1e-3f)
        val med = ((pct(sl, 0.5) - lo) / span).coerceIn(0.01f, 0.99f)
        val tmed = (REF_MED - REF_LO) / (REF_HI - REF_LO)
        val g = (ln(tmed) / ln(med)).coerceIn(0.8f, 1.25f)
        for (i in 0 until n) {
            val ln01 = ((l[i] - lo) / span).coerceIn(0f, 1f)
            val target = REF_LO + ln01.pow(g) * (REF_HI - REF_LO)
            l[i] += strength * (target - l[i])
        }

        // 2. white balance: clean warm-white highlights (removes pink/blue casts on white clothes, walls)
        var sa = 0.0
        var sb = 0.0
        var cnt = 0
        for (i in 0 until n) if (l[i] > 72f) { sa += a[i]; sb += b[i]; cnt++ }
        if (cnt > 200) {
            val da = (REF_HI_A - sa / cnt).toFloat()
            val db = (REF_HI_B - sb / cnt).toFloat()
            for (i in 0 until n) {
                val w = ((l[i] - 35f) / 45f).coerceIn(0f, 1f) * 0.8f * strength
                a[i] += da * w; b[i] += db * w
            }
        }

        // 3. vibrance towards the reference's natural colourfulness
        var midCount = 0
        for (i in 0 until n) if (l[i] > 20f && l[i] < 88f) midCount++
        val cm = if (midCount > 0) {
            val c = sample(n) { i -> if (l[i] > 20f && l[i] < 88f) hypot(a[i], b[i]) else Float.NaN }
                .filter { !it.isNaN() }.toFloatArray()
            if (c.isEmpty()) 10f else { c.sort(); pct(c, 0.5) }
        } else 10f
        // gentle: lifts dull colour, calms over-coloured results, never pushes strong colour further
        var k = (REF_CHROMA / max(cm, 1e-3f)).coerceIn(0.65f, 1.35f)
        k = 1f + (k - 1f) * strength
        for (i in 0 until n) {
            val c = hypot(a[i], b[i])
            val boost = 1f + (k - 1f) * (1f - c / 30f).coerceIn(0f, 1f)
            a[i] *= boost; b[i] *= boost
            // natural-colour limiter: soft ceiling on very strong colour (tames AI over-colouring)
            val c2 = c * boost
            if (c2 > 28f) {
                val lim = (28f + 14f * tanh((c2 - 28f) / 14f)) / c2
                a[i] *= lim; b[i] *= lim
            }
        }
    }

    // Overall colour balance of the reference photo (red/green and blue/green of its normal pixels)
    private const val REF_RATIO_R = 1.091f
    private const val REF_RATIO_B = 0.956f

    /**
     * Automatic white balance towards the reference photo: removes overall orange/pink/blue casts
     * (scales red and blue, so skin stays warm while walls and whites become normal).
     */
    private fun whiteBalance(img: LabImage, strength: Float) {
        val n = img.L.size
        val all = FloatArray(n * 3)
        for (i in 0 until n) {
            all[3 * i] = img.L[i].coerceIn(0f, 100f); all[3 * i + 1] = img.a[i]; all[3 * i + 2] = img.b[i]
        }
        val lab = Mat(img.h, img.w, CvType.CV_32FC3)
        lab.put(0, 0, all)
        val rgb = Mat()
        Imgproc.cvtColor(lab, rgb, Imgproc.COLOR_Lab2RGB)
        rgb.get(0, 0, all)

        // robust grey-world: average of normal pixels (not too dark/bright, not vivid)
        var sr = 0.0
        var sg = 0.0
        var sbl = 0.0
        var cnt = 0
        for (i in 0 until n) {
            val l = img.L[i]
            if (l > 20f && l < 95f && hypot(img.a[i], img.b[i]) < 50f) {
                sr += all[3 * i].coerceIn(0f, 1f); sg += all[3 * i + 1].coerceIn(0f, 1f); sbl += all[3 * i + 2].coerceIn(0f, 1f)
                cnt++
            }
        }
        if (cnt < 100 || sg <= 0.0) { release(lab, rgb); return }
        val gR = (REF_RATIO_R / (sr / sg).toFloat()).pow(strength).coerceIn(0.6f, 1.6f)
        val gB = (REF_RATIO_B / (sbl / sg).toFloat()).pow(strength).coerceIn(0.6f, 1.6f)

        var before = 0.0
        var after = 0.0
        for (i in 0 until n) {
            val r = all[3 * i].coerceIn(0f, 1f)
            val g = all[3 * i + 1].coerceIn(0f, 1f)
            val b = all[3 * i + 2].coerceIn(0f, 1f)
            before += r + g + b
            all[3 * i] = (r * gR).coerceIn(0f, 1f)
            all[3 * i + 1] = g
            all[3 * i + 2] = (b * gB).coerceIn(0f, 1f)
            after += all[3 * i] + all[3 * i + 1] + all[3 * i + 2]
        }
        val keep = if (after > 0) (before / after).toFloat() else 1f     // same overall brightness
        for (j in all.indices) all[j] = (all[j] * keep).coerceIn(0f, 1f)

        rgb.put(0, 0, all)
        Imgproc.cvtColor(rgb, lab, Imgproc.COLOR_RGB2Lab)
        lab.get(0, 0, all)
        for (i in 0 until n) {
            img.L[i] = all[3 * i]; img.a[i] = all[3 * i + 1]; img.b[i] = all[3 * i + 2]
        }
        release(lab, rgb)
    }

    private inline fun sample(n: Int, f: (Int) -> Float): FloatArray {
        val step = max(1, n / 60000)
        val out = FloatArray((n + step - 1) / step)
        var j = 0
        var i = 0
        while (i < n) { out[j++] = f(i); i += step }
        return out
    }

    private fun pct(sorted: FloatArray, p: Double): Float =
        sorted[((sorted.size - 1) * p).roundToInt()]

    // ---------------------------------------------------------------- 922 x 2048 framing

    const val RATIO = 922.0 / 2048.0

    /** Largest 922:2048 box inside a w x h photo, kept around the faces. */
    fun cropBox(w: Int, h: Int, face: PointF?): Rect {
        return if (w.toDouble() / h > RATIO) {
            val cw = (h * RATIO).roundToInt().coerceIn(1, w)
            val cx = face?.x?.toDouble() ?: (w / 2.0)
            val x = (cx - cw / 2.0).roundToInt().coerceIn(0, w - cw)
            Rect(x, 0, cw, h)
        } else {
            val ch = (w / RATIO).roundToInt().coerceIn(1, h)
            val cy = face?.let { it.y + ch * 0.15 } ?: (h / 2.0)
            val y = (cy - ch / 2.0).roundToInt().coerceIn(0, h - ch)
            Rect(0, y, w, ch)
        }
    }

    /** Soft blurred, darkened copy of the photo filling W x H (behind a "fit whole" image). */
    fun blurredBackground(rgb: Mat, W: Int, H: Int): Mat {
        val w = rgb.cols()
        val h = rgb.rows()
        val sb = max(W.toDouble() / w, H.toDouble() / h)
        val small = Mat()
        Imgproc.resize(rgb, small, Size(max(1.0, w * sb / 8), max(1.0, h * sb / 8)), 0.0, 0.0, Imgproc.INTER_AREA)
        Imgproc.GaussianBlur(small, small, Size(0.0, 0.0), 6.0)
        val big = Mat()
        Imgproc.resize(small, big, Size(ceil(w * sb) + 2, ceil(h * sb) + 2), 0.0, 0.0, Imgproc.INTER_LINEAR)
        val bx = (big.cols() - W) / 2
        val by = (big.rows() - H) / 2
        val roi = big.submat(by, by + H, bx, bx + W)
        val out = Mat()
        roi.convertTo(out, -1, 0.55, 0.0)
        release(small, roi, big)
        return out
    }
}
