# modesty-blur (prototype)

Live camera filter: every person the system believes is female is replaced by a smooth,
full-silhouette blur (not a box). Family members you enroll are shown normally (desktop build).
Runs fully on-device. Built to be tested on webcam, recorded video and an Android phone.

## Downloads (built automatically)

Every push to `main` builds all four apps and publishes a new GitHub Release (`v0.1.<run number>`):

| Platform | File | Notes |
|---|---|---|
| Android | `modesty-blur-android.apk` | Phone camera, on-device models, in-app updater |
| Windows | `modesty-blur-windows.zip` | unzip, run `modesty-blur.exe` |
| macOS | `modesty-blur-macos.zip` | unzip, run `./modesty-blur` from Terminal (allow camera for Terminal) |
| Linux | `modesty-blur-linux.zip` | unzip, run `./modesty-blur` |

Desktop builds are CPU-only and download the CLIP weights on first launch (internet needed once).
For GPU speed, run from source (below).

### Android updates
The app checks the Releases page on launch (about once an hour), shows an **Install / Later** banner when
a newer APK exists, and Settings always shows the pending update plus a "Check now" button and a switch
to turn automatic checks off. This requires the repository (or at least its Releases) to be **public**.
Android asks once for permission to install from this app.

The APK is signed with a prototype key committed at `android/app/prototype-release.p12`, so every build
installs as an update over the previous one. Move to GitHub secrets before sharing it widely.

### Android app: clean mode, debug mode, family faces
- Normal mode is just the blurred camera feed. Two small buttons (flip, settings) appear when you tap the
  screen and fade after a few seconds. No text.
- Settings > Debug overlay labels every person with what the app decided (blur / clear / family), the
  probability, and what it is doing right now (classifying, face check, idle), plus fps and timings.
- Settings > Family faces: pick photos of each family member. Only the numeric face data stays on the
  phone. A family member is un-blurred only after several consecutive face matches and is re-verified
  every few seconds; an unseen or unrecognized face stays blurred.
- Settings > Speed and quality: Fast (small model, 640x480), Balanced (default), Quality.

## How it works (fail-closed)

1. YOLO11-seg finds every person and gives an exact body outline; a tracker gives each a stable ID.
2. A new person is blurred immediately, before any classification.
3. CLIP estimates P(female) every few frames (smoothed per person).
4. A person is un-blurred only if (a) confirmed family by face match (desktop and Android), or
   (b) repeatedly and confidently classified as not female. Any uncertainty keeps the blur on.
5. If the detector briefly loses a blurred person, the last silhouette keeps blurring for ~10 frames.
6. On Android, if a frame fails to process, the screen shows black instead of the raw camera.

## Run from source (desktop)

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m modesty_blur run --source 0 --debug
python -m modesty_blur run --source street.mp4 --save out.mp4 --headless
python -m modesty_blur enroll family/ --out family_gallery.npz    # family/<name>/*.jpg
```

## Test

```bash
python tests/test_core.py                       # policy + blur checks, no models needed
python -m modesty_blur selftest                 # loads every model, runs blank frames (CI does this per OS)
python tools/eval_classifier.py eval_set/       # eval_set/woman/*.jpg, eval_set/man/*.jpg
```
The number to watch is the **miss rate** (a woman left unblurred). Lower `blur_threshold` until it is
near zero; the price is more false blurs. On Android the same knob is Settings > Blur sensitivity.

## Repository layout

```
modesty_blur/        Python pipeline (desktop)
desktop_entry.py     PyInstaller entry point
android/             Kotlin app (CameraX + ONNX Runtime), same decision policy
tools/export_models.py   exports YOLO + CLIP to ONNX for the Android app (run by CI)
.github/workflows/build-release.yml   build + release on every push
```

## Known limits

- Appearance-based classification makes mistakes (distance, angle, clothing, lighting). It is a
  best-effort filter, not a guarantee.
- "Married woman" cannot be reliably seen, so it is not attempted.
- Android: new people stay blurred for roughly a second until the background models clear them.
  Speed depends on the phone; Settings > Speed and quality has Fast / Balanced / Quality presets.
- Avatar face / lip-sync overlay and the glasses version are later phases.
