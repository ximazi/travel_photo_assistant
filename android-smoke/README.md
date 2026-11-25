# Android CameraX local smoke spike

Companion to issue #4 / draft #14. This is an isolated test application, not a product client. No backend, credentials, Gemini, human images, nor INTERNET permission. Use only a synthetic virtual scene or a printed/test target. Do not use a host webcam.

## Build
Windows PowerShell, Android SDK with API 36, JDK 21, Gradle 8.14.3. From this directory:

```powershell
$env:JAVA_HOME = "$HOME\.jdks\jbr-21.0.11"
$env:ANDROID_HOME = "$HOME\AppData\Local\Android\Sdk"
gradle assembleDebug
```

The debug APK is at `app/build/outputs/apk/debug/app-debug.apk`. AGP 8.13.0 and CameraX 1.4.2 are fetched from Google Maven when missing. Check packaged manifest with SDK `aapt dump permissions`. Do not publish the APK.

The pure Java gate is independently testable without Android dependencies:

```powershell
$classes = Join-Path $env:TEMP 'tpa-smoke-test-classes'
New-Item -ItemType Directory -Path $classes -Force | Out-Null
& "$env:JAVA_HOME\bin\javac.exe" -d $classes app/src/main/java/com/travelphoto/smoke/ReviewGate.java tools/ReviewGateTest.java
& "$env:JAVA_HOME\bin\java.exe" -cp $classes com.travelphoto.smoke.ReviewGateTest
```

## Emulator or test device
Use an explicitly selected serial. For emulator launch, use `-read-only -no-snapshot -camera-back virtualscene -camera-front emulated`; never change Windows features for this spike. With permission on the test device, install and launch the APK. Capture logs using `adb -s <serial> logcat -s TpaSmoke`.

- Three use cases share CameraX ViewPort. The fixed unmirrored, upright crop probe (.25,.35) uses CameraX source-to-PreviewView mapping. CameraX handles front display mirroring; the app does not mirror again. This is a **known coordinate**, not an automatically detected target. Move a synthetic asymmetric target so its off-center feature aligns with the probe; retain screenshots plus crop/rotation/preview coordinates across portrait and landscape and for front camera. Emulator front feed may use a different image.
- **Review normal:** starts a 5 s deadline at button tap, latches the first later analyzed sequence in the current camera/viewport version and responds after 900 ms. Green denotes local stub acceptance only.
- While reviewing, rotate or toggle zoom: old ticket/overlay invalidates; old response logs discarded and cannot set green. A fresh request is needed. Camera switch and background also invalidate.
- **Review timeout:** stub response comes only after the separate 5 s deadline; it must be discarded.
- **Offline stub:** local simulated failure; no actual network is attempted. Test manual shutter after timeout and while this stub reports offline. It remains independent of review. Capture writes into app-private cache and checks JPEG MIME, recorded dimensions and pixel decode from the result. Do not export raw frames or commit logs containing photos.
- Actual physical lens identifier remains unknown. Stock emulator rotation/mirror observations are not physical phone acceptance. No automatic subject tracking, user-facing photography grading, real remote latency, or product release claim.

Logger labels include session/bind number, version, sequence, sensor-to-target rotation, applied crop, front flag, source/display probe, monotonic timing, fresh review ticket/floor, result sequence and shutter evidence.

