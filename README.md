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

## How it works (fail-closed)

1. YOLO11-seg finds every person and gives an exact body outline; a tracker gives each a stable ID.
2. A new person is blurred immediately, before any classification.
3. CLIP estimates P(female) every few frames (smoothed per person).
4. A person is un-blurred only if (a) confirmed family by face match (desktop only for now), or
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
- Android v0.1 has no family whitelist yet and classifies one person per frame, so a new person stays
  blurred for a moment before they can be cleared. Speed depends on the phone.
- Avatar face / lip-sync overlay and the glasses version are later phases.
