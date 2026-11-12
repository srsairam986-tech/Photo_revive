package com.sairam.photorevive

import android.annotation.SuppressLint
import android.content.ContentValues
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.provider.MediaStore
import android.view.MotionEvent
import android.view.View
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.IntentCompat
import androidx.exifinterface.media.ExifInterface
import androidx.lifecycle.lifecycleScope
import com.sairam.photorevive.databinding.ActivityMainBinding
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.opencv.android.OpenCVLoader
import java.io.File
import java.io.FileOutputStream
import kotlin.math.max
import kotlin.math.min

class MainActivity : AppCompatActivity() {

    private lateinit var ui: ActivityMainBinding
    private var source: Bitmap? = null
    private var result: Bitmap? = null
    private var shown: Bitmap? = null
    private var busy = false

    private val pick = registerForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        if (uri != null) load(uri)
    }

    @SuppressLint("ClickableViewAccessibility")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        ui = ActivityMainBinding.inflate(layoutInflater)
        setContentView(ui.root)

        if (!OpenCVLoader.initLocal()) toast("Image engine failed to load")

        ui.btnOpen.setOnClickListener { if (!busy) pick.launch("image/*") }
        ui.btnRevive.setOnClickListener { revive() }
        ui.btnSave.setOnClickListener { save() }
        ui.viewer.setOnClickListener {
            ui.panel.visibility = if (ui.panel.visibility == View.VISIBLE) View.GONE else View.VISIBLE
        }
        // hold to see the original, release to see the result
        ui.btnCompare.setOnTouchListener { v, e ->
            when (e.action) {
                MotionEvent.ACTION_DOWN -> source?.let { ui.viewer.setImageBitmap(it) }
                MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                    shown?.let { ui.viewer.setImageBitmap(it) }
                    v.performClick()
                }
            }
            true
        }
        handleIntent(intent)
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        handleIntent(intent)
    }

    private fun handleIntent(intent: Intent?) {
        val uri = when (intent?.action) {
            Intent.ACTION_VIEW -> intent.data
            Intent.ACTION_SEND -> IntentCompat.getParcelableExtra(intent, Intent.EXTRA_STREAM, Uri::class.java)
            else -> null
        }
        if (uri != null) load(uri)
    }

    private fun load(uri: Uri) {
        lifecycleScope.launch {
            try {
                val bmp = withContext(Dispatchers.IO) { decode(uri) }
                source = bmp
                result = null
                shown = bmp
                ui.viewer.setImageBitmap(bmp)
                ui.hint.visibility = View.GONE
                ui.status.text = "${bmp.width}×${bmp.height} · tap Colourise"
            } catch (e: Throwable) {
                toast("Can't open this image: ${e.message}")
            }
        }
    }

    private fun decode(uri: Uri): Bitmap {
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        contentResolver.openInputStream(uri)!!.use { BitmapFactory.decodeStream(it, null, bounds) }
        var sample = 1
        while (max(bounds.outWidth, bounds.outHeight) / (sample * 2) >= 2400) sample *= 2
        val opts = BitmapFactory.Options().apply {
            inSampleSize = sample
            inPreferredConfig = Bitmap.Config.ARGB_8888
        }
        val bmp = contentResolver.openInputStream(uri)!!.use { BitmapFactory.decodeStream(it, null, opts) }
            ?: error("unsupported format")
        val orientation = try {
            contentResolver.openInputStream(uri)!!.use {
                ExifInterface(it).getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL)
            }
        } catch (e: Exception) {
            ExifInterface.ORIENTATION_NORMAL
        }
        val degrees = when (orientation) {
            ExifInterface.ORIENTATION_ROTATE_90 -> 90f
            ExifInterface.ORIENTATION_ROTATE_180 -> 180f
            ExifInterface.ORIENTATION_ROTATE_270 -> 270f
            else -> 0f
        }
        if (degrees == 0f) return bmp
        val rotated = Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, Matrix().apply { postRotate(degrees) }, true)
        if (rotated !== bmp) bmp.recycle()
        return rotated
    }

    private fun revive() {
        val src = source ?: return toast("Open a photo first")
        if (busy) return
        val opt = ReviveOptions(
            framing = if (ui.groupFrame.checkedButtonId == R.id.btnFit) Framing.FIT else Framing.FILL,
            eightK = ui.groupQuality.checkedButtonId == R.id.btn8k,
            strongClean = ui.groupClean.checkedButtonId == R.id.btnStrong,
        )
        setBusy(true)
        val started = System.currentTimeMillis()
        lifecycleScope.launch {
            try {
                val out = withContext(Dispatchers.Default) {
                    Reviver(applicationContext).run(src, opt,
                        progress = { f, msg ->
                            runOnUiThread {
                                ui.progress.setProgressCompat((f * 100).toInt(), true)
                                ui.status.text = msg
                            }
                        },
                        preview = { p -> runOnUiThread { shown = p; ui.viewer.setImageBitmap(p) } })
                }
                result = out
                shown = screenCopy(out)
                ui.viewer.setImageBitmap(shown)
                val secs = (System.currentTimeMillis() - started) / 1000
                ui.status.text = "Done · ${out.width}×${out.height} in ${secs}s · tap Save"
            } catch (e: OutOfMemoryError) {
                ui.status.text = "Not enough memory — close other apps or use 4K"
            } catch (e: Throwable) {
                ui.status.text = "Failed: ${e.message}"
            } finally {
                setBusy(false)
            }
        }
    }

    /** 8K pictures are too big for the screen to draw, so the viewer shows a 4K copy; Save keeps the full size. */
    private fun screenCopy(bmp: Bitmap): Bitmap {
        val s = min(1f, 1728f / bmp.width)
        return if (s >= 1f) bmp
        else Bitmap.createScaledBitmap(bmp, (bmp.width * s).toInt(), (bmp.height * s).toInt(), true)
    }

    private fun save() {
        val bmp = result ?: return toast("Colourise a photo first")
        if (busy) return
        setBusy(true)
        ui.status.text = "Saving…"
        lifecycleScope.launch {
            val name = "PhotoRevive_${System.currentTimeMillis()}.jpg"
            val ok = withContext(Dispatchers.IO) { runCatching { writeJpeg(bmp, name) }.isSuccess }
            setBusy(false)
            ui.status.text = if (ok) "Saved to Pictures/PhotoRevive" else "Could not save"
        }
    }

    private fun writeJpeg(bmp: Bitmap, name: String) {
        if (Build.VERSION.SDK_INT >= 29) {
            val values = ContentValues().apply {
                put(MediaStore.Images.Media.DISPLAY_NAME, name)
                put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
                put(MediaStore.Images.Media.RELATIVE_PATH, Environment.DIRECTORY_PICTURES + "/PhotoRevive")
                put(MediaStore.Images.Media.IS_PENDING, 1)
            }
            val uri = contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values)!!
            contentResolver.openOutputStream(uri)!!.use { bmp.compress(Bitmap.CompressFormat.JPEG, 96, it) }
            values.clear()
            values.put(MediaStore.Images.Media.IS_PENDING, 0)
            contentResolver.update(uri, values, null, null)
        } else {
            @Suppress("DEPRECATION")
            val dir = File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES), "PhotoRevive")
            dir.mkdirs()
            FileOutputStream(File(dir, name)).use { bmp.compress(Bitmap.CompressFormat.JPEG, 96, it) }
        }
    }

    private fun setBusy(b: Boolean) {
        busy = b
        ui.progress.visibility = if (b) View.VISIBLE else View.INVISIBLE
        if (b) ui.progress.setProgressCompat(0, false)
        listOf(ui.btnOpen, ui.btnRevive, ui.btnSave).forEach { it.isEnabled = !b }
        ui.groupFrame.isEnabled = !b
        ui.groupQuality.isEnabled = !b
        ui.groupClean.isEnabled = !b
    }

    private fun toast(msg: String) = Toast.makeText(this, msg, Toast.LENGTH_SHORT).show()
}
