package com.travelphoto.smoke;

import android.Manifest;
import android.content.pm.PackageManager;
import android.content.res.Configuration;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Rect;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.os.SystemClock;
import android.util.Log;
import android.view.View;
import android.widget.*;
import androidx.activity.ComponentActivity;
import androidx.camera.core.*;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.camera.view.transform.CoordinateTransform;
import androidx.camera.view.transform.ImageProxyTransformFactory;
import androidx.camera.view.transform.OutputTransform;
import androidx.core.content.ContextCompat;
import java.io.File;
import java.io.IOException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicLong;

public final class MainActivity extends ComponentActivity {
    private final Handler ui = new Handler(Looper.getMainLooper());
    private final ExecutorService analysisExecutor = Executors.newSingleThreadExecutor();
    private final AtomicLong frames = new AtomicLong();
    private final ReviewGate gate = new ReviewGate();
    private PreviewView previewView;
    private Probe overlay;
    private TextView status;
    private ProcessCameraProvider provider;
    private ImageCapture capture;
    private Camera camera;
    private boolean front;
    private boolean zoomed;
    private long latest = -1;
    private long session = -1;
    private long lastLoggedMs;
    private long frameFloor;
    private boolean awaitingZoom;
    private boolean foreground;
    private long stubDelayMs;

    @Override public void onCreate(Bundle saved) {
        super.onCreate(saved);
        LinearLayout column = new LinearLayout(this);
        column.setOrientation(LinearLayout.VERTICAL);
        status = new TextView(this);
        // Status changes must not resize the preview and invalidate in-flight reviews.
        status.setSingleLine(true);
        status.setEllipsize(android.text.TextUtils.TruncateAt.END);
        status.setText("CameraX local smoke | physical lens unknown");
        column.addView(status);
        FrameLayout pane = new FrameLayout(this);
        previewView = new PreviewView(this);
        previewView.setImplementationMode(PreviewView.ImplementationMode.COMPATIBLE);
        previewView.setScaleType(PreviewView.ScaleType.FILL_CENTER);
        pane.addView(previewView, new FrameLayout.LayoutParams(-1, -1));
        overlay = new Probe();
        pane.addView(overlay, new FrameLayout.LayoutParams(-1, -1));
        column.addView(pane, new LinearLayout.LayoutParams(-1, 0, 1));
        LinearLayout reviewRow = new LinearLayout(this);
        button(reviewRow, "Review normal", () -> review(900));
        button(reviewRow, "Review timeout", () -> review(6000));
        button(reviewRow, "Offline stub", this::offline);
        column.addView(reviewRow);
        LinearLayout cameraRow = new LinearLayout(this);
        button(cameraRow, "Switch camera", () -> { front = !front; zoomed = false; rebind("camera-switch"); });
        button(cameraRow, "Toggle zoom", this::zoom);
        button(cameraRow, "Manual shutter", this::takePhoto);
        column.addView(cameraRow);
        setContentView(column);
        column.setOnApplyWindowInsetsListener((v, insets) -> {
            v.setPadding(insets.getSystemWindowInsetLeft(), insets.getSystemWindowInsetTop(),
                    insets.getSystemWindowInsetRight(), insets.getSystemWindowInsetBottom());
            return insets;
        });

        previewView.addOnLayoutChangeListener((v, l, t, r, b, ol, ot, or, ob) -> {
            if (r-l != or-ol || b-t != ob-ot) rebind("viewport-size");
        });
        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED)
            requestPermissions(new String[]{Manifest.permission.CAMERA}, 1);
        else openCamera();
    }

    private void button(LinearLayout row, String title, Runnable f) {
        Button b = new Button(this);
        b.setAllCaps(false);
        b.setText(title);
        b.setMinHeight(0);
        b.setMinimumHeight(0);
        b.setOnClickListener(v -> f.run());
        row.addView(b, new LinearLayout.LayoutParams(0, -2, 1));
    }

    @Override public void onRequestPermissionsResult(int code, String[] p, int[] results) {
        super.onRequestPermissionsResult(code, p, results);
        if (code == 1 && results.length > 0 && results[0] == PackageManager.PERMISSION_GRANTED) openCamera();
        else say("camera permission denied");
    }

    private void openCamera() {
        var future = ProcessCameraProvider.getInstance(this);
        future.addListener(() -> {
            if (isDestroyed()) return;
            try { provider = future.get(); rebind("provider-ready"); }
            catch (Exception e) { say("provider failed: " + e.getClass().getSimpleName()); }
        }, ContextCompat.getMainExecutor(this));
    }

    @Override protected void onStart() { super.onStart(); foreground = true; }

    @Override protected void onStop() {
        foreground = false;
        invalidate("background");
        super.onStop();
    }

    @Override public void onConfigurationChanged(Configuration c) {
        super.onConfigurationChanged(c);
        session++; // Drop analysis callbacks from the old display before rebind.
        invalidate("orientation");
        previewView.post(() -> rebind("orientation-layout"));
    }

    private void invalidate(String reason) {
        gate.invalidate();
        latest = -1;
        frameFloor = frames.get();
        overlay.clear();
        say("invalidated: " + reason + " v=" + gate.version() + " lens=unknown");
    }

    private void rebind(String reason) {
        invalidate(reason);
        if (provider == null || previewView.getDisplay() == null || previewView.getWidth() == 0) return;
        ViewPort viewport = previewView.getViewPort();
        if (viewport == null) return;
        int rotation = previewView.getDisplay().getRotation();
        long bind = ++session;
        capture = null;
        camera = null;
        awaitingZoom = false;
        Preview preview = new Preview.Builder().setTargetRotation(rotation).build();
        ImageAnalysis analysis = new ImageAnalysis.Builder().setTargetRotation(rotation)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST).build();
        ImageCapture nextCapture = new ImageCapture.Builder().setTargetRotation(rotation).build();
        ImageProxyTransformFactory factory = new ImageProxyTransformFactory();
        factory.setUsingCropRect(true);
        factory.setUsingRotationDegrees(true);
        analysis.setAnalyzer(analysisExecutor, image -> {
            try {
                long seq = frames.incrementAndGet();
                Rect crop = new Rect(image.getCropRect());
                int degrees = image.getImageInfo().getRotationDegrees();
                OutputTransform source = factory.getOutputTransform(image);
                int w = degrees % 180 == 0 ? crop.width() : crop.height();
                int h = degrees % 180 == 0 ? crop.height() : crop.width();
                ui.post(() -> onFrame(bind, seq, crop, degrees, w, h, source));
            } finally { image.close(); }
        });
        preview.setSurfaceProvider(previewView.getSurfaceProvider());
        CameraSelector selector = front ? CameraSelector.DEFAULT_FRONT_CAMERA : CameraSelector.DEFAULT_BACK_CAMERA;
        try {
            provider.unbindAll();
            camera = provider.bindToLifecycle(this, selector, new UseCaseGroup.Builder().setViewPort(viewport)
                    .addUseCase(preview).addUseCase(analysis).addUseCase(nextCapture).build());
            capture = nextCapture;
            zoomed = false;
        } catch (RuntimeException e) { say("camera binding failed: " + e.getClass().getSimpleName()); }
    }

    private void onFrame(long bind, long seq, Rect crop, int degrees, int width, int height, OutputTransform source) {
        if (!foreground || isDestroyed() || bind != session || camera == null || awaitingZoom || seq <= frameFloor) return;
        OutputTransform target = previewView.getOutputTransform();
        if (target == null || previewView.getPreviewStreamState().getValue() != PreviewView.StreamState.STREAMING) return;
        float[] point = {0.25f * width, 0.35f * height};
        try { new CoordinateTransform(source, target).mapPoints(point); }
        catch (RuntimeException e) { return; }
        latest = seq;
        overlay.x = point[0]; overlay.y = point[1]; overlay.probe = true; overlay.invalidate();
        long now = SystemClock.elapsedRealtime();
        var ticket = gate.latch(seq, now);
        if (ticket != null) {
            Log.i("TpaSmoke", "t=" + now + " review frame selected: " + seq + " " + ticket);
            ui.postDelayed(() -> finishReview(ticket, seq), stubDelayMs);
        }
        if (now - lastLoggedMs >= 1500) {
            lastLoggedMs = now;
            Log.i("TpaSmoke", "frame=" + seq + " session=" + bind + " version=" + gate.version() +
                    " crop=" + crop.toShortString() + " rotation=" + degrees + " front=" + front +
                    " uprightProbe=.25,.35 screen=" + point[0] + "," + point[1] + " lens=unknown");
        }
    }

    private void review(long delay) {
        ReviewGate.Ticket ticket = gate.begin(latest < 0 ? -1 : frames.get(), SystemClock.elapsedRealtime(), 5000);
        if (ticket == null) { say("review not started: no fresh frame or already pending"); return; }
        overlay.reviewed = false;
        overlay.invalidate();
        stubDelayMs = delay;
        say("review started: " + ticket);
        ui.postDelayed(() -> {
            if (gate.expire(ticket, SystemClock.elapsedRealtime())) say("review timeout: " + ticket + "; manual shutter still available");
        }, 5000);
    }

    private void finishReview(ReviewGate.Ticket ticket, long resultFrame) {
        if (gate.accept(ticket, resultFrame, SystemClock.elapsedRealtime())) {
            overlay.reviewed = true;
            overlay.invalidate();
            say("ready (stub only): fresh frame " + resultFrame + " " + ticket);
        } else Log.i("TpaSmoke", "discarded reply: " + ticket + " resultFrame=" + resultFrame + " currentVersion=" + gate.version());
    }

    private void offline() {
        invalidate("simulated offline");
        say("offline stub; manual shutter still available");
    }

    private void zoom() {
        if (awaitingZoom || camera == null || camera.getCameraInfo().getZoomState().getValue() == null) return;
        invalidate("zoom-change");
        awaitingZoom = true;
        var z = camera.getCameraInfo().getZoomState().getValue();
        zoomed = !zoomed;
        Camera target = camera;
        var applied = target.getCameraControl().setZoomRatio(zoomed ? Math.min(2, z.getMaxZoomRatio()) : 1);
        applied.addListener(() -> {
            if (camera != target || isDestroyed()) return;
            awaitingZoom = false;
            frameFloor = frames.get();
            try { applied.get(); say("zoom applied; fresh local frame required"); }
            catch (Exception e) { say("zoom failed: " + e.getClass().getSimpleName()); }
        }, ContextCompat.getMainExecutor(this));
    }

    private void takePhoto() {
        ImageCapture c = capture;
        if (c == null) { say("shutter unavailable: camera not bound"); return; }
        try {
            File f = File.createTempFile("tpa-smoke-", ".jpg", getCacheDir());
            c.takePicture(new ImageCapture.OutputFileOptions.Builder(f).build(), ContextCompat.getMainExecutor(this),
                    new ImageCapture.OnImageSavedCallback() {
                        @Override public void onImageSaved(ImageCapture.OutputFileResults ignored) {
                            BitmapFactory.Options bounds = new BitmapFactory.Options();
                            bounds.inJustDecodeBounds = true;
                            BitmapFactory.decodeFile(f.getPath(), bounds);
                            BitmapFactory.Options pixels = new BitmapFactory.Options();
                            pixels.inSampleSize = 4;
                            Bitmap decoded = BitmapFactory.decodeFile(f.getPath(), pixels);
                            boolean jpeg = "image/jpeg".equals(bounds.outMimeType) && decoded != null;
                            if (decoded != null) decoded.recycle();
                            say("shutter jpeg=" + jpeg + " " + bounds.outWidth + "x" + bounds.outHeight +
                                    " bytes=" + f.length() + " private=" + f.getName());
                        }
                        @Override public void onError(ImageCaptureException e) { say("shutter failed code=" + e.getImageCaptureError()); }
                    });
        } catch (IOException e) { say("shutter local file error"); }
    }

    private void say(String s) {
        status.setText(s);
        Log.i("TpaSmoke", "t=" + SystemClock.elapsedRealtime() + " " + s);
    }

    @Override protected void onDestroy() {
        ui.removeCallbacksAndMessages(null);
        if (provider != null) provider.unbindAll();
        analysisExecutor.shutdown();
        super.onDestroy();
    }

    private final class Probe extends View {
        private final Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG);
        float x, y;
        boolean probe, reviewed;
        Probe() { super(MainActivity.this); }
        void clear() { probe = false; reviewed = false; invalidate(); }
        @Override protected void onDraw(Canvas canvas) {
            if (!probe) return;
            paint.setColor(reviewed ? Color.GREEN : Color.YELLOW);
            paint.setStrokeWidth(4);
            canvas.drawLine(x - 16, y, x + 16, y, paint);
            canvas.drawLine(x, y - 16, x, y + 16, paint);
        }
    }
}

