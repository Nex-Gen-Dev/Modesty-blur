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
- Settings > Speed and quality: Fast, Balanced (default), Quality and 1080p. All are 16:9 shown in portrait.
- Settings > Person detection: confidence needed to count something as a person (raise it if feet or floor
  patches get blurred).
- Settings > Test lab: pick photos from the gallery, say whether they show women or men, and get the
  decisions plus a MISS-rate summary, without anyone walking in front of the camera.
- Family enrollment accepts photos or a short video of the person turning their head. Each photo is stored
  four ways (normal, mirrored, tilted each way) and video keeps only samples with a new angle, so the person
  does not have to look at the camera or smile.

## How it works (fail-closed)

1. YOLO11-seg finds every person and gives an exact body outline; a tracker gives each a stable ID.
2. A new person is blurred immediately, before any classification.
3. CLIP estimates P(female) every few frames (smoothed per person).
4. A person is un-blurred only if (a) confirmed family by face match (desktop and Android), or
   (b) repeatedly and confidently classified as not female. Any uncertainty keeps the blur on.
5. Camera motion is measured every frame (a few ms), and every tracked silhouette is shifted with the scene, so
   blur follows a panning camera and people keep their ID. If the detector briefly loses a blurred person, the
   shifted silhouette keeps blurring for at most 350 ms, and is dropped as soon as a live detection covers it.
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

## Teach it with your own pictures (how to contribute)

Out of the box the app guesses with CLIP (a general model, about half a second per person) and with a
face-gender model that is only ever allowed to ADD blur. Both are fine for a prototype and both make
mistakes. The fix is a small model trained on pictures from your world (Jerusalem streets, the clothing
you actually see, the camera height of the glasses). It runs in a few milliseconds, so decisions become
near-instant, and it is the only way to get the miss rate close to zero.

**What to collect** (privately, with consent where people are identifiable; never upload or share it)
- Short videos or photos walking around, filmed at chest or eye height like glasses would.
- Cover the variety on purpose: women with hats, berets, scarves, wigs, long skirts, long sleeves, from the
  front, side and back, far away and close, in shade and sun, with strollers, in groups. Men with long coats,
  black hats, kippot, beards, long hair, backpacks. Girls and boys. Bad angles and partial views are the
  most valuable pictures.
- Aim for at least 300 crops per class to start and 1500+ for a solid model. More of the hard cases beats more
  of the easy ones.
- For bulk, public pedestrian datasets with gender labels (PA-100K, PETA, RAP) are the usual starting point,
  but they are research-licensed and mostly not from your setting, so use them to bootstrap, not instead of
  your own footage.

**What the classes are.** Only things a camera can see: woman, man, girl (child), boy (child), teen girl,
teen boy. Women, girls and teen girls are blurred. The model cannot and should not be taught anybody's religion,
nationality or ethnicity: none of that is visible in a picture, so labels like "Jewish woman" would only teach it
stereotypes (and would make it worse at the actual job). Anything it should see differently because of clothing
(a hat, a scarf) is handled by showing it many examples of each visible class wearing those things.

**Big datasets (tens of thousands of pictures): use Google Colab.** Put the pictures in your Google Drive
(one folder per class, or zip files per class) and open
[the training notebook](https://colab.research.google.com/github/Nex-Gen-Dev/Modesty-blur/blob/main/tools/train_colab.ipynb).
It trains on a free GPU, saves `person_cls.onnx` + `person_cls.json` to your Drive, and you commit them to `models/`.

**The workflow** (on your computer, Python installed)
```bash
pip install -r requirements.txt
python tools/extract_crops.py my_footage/ --out dataset/unlabeled      # 1. cut people out of videos/photos
python tools/label.py dataset/unlabeled --out dataset                   # 2. sort them: w m g b t y x (one key each)
python tools/train_person_cls.py dataset --out models --female-weight 3 # 3. train, see the miss-rate table
git add models/person_cls.onnx models/person_cls.json && git commit -m "trained classifier" && git push
```
The next build bundles the model automatically and the app switches from CLIP to it (the debug overlay shows
"classifier: trained model"). Re-run steps 2 and 3 whenever the Test lab or the street shows you a miss: add
those pictures, retrain, push. If you are not comfortable running this yourself, send me the `dataset/` folder
layout (never the photos through public channels) or a few labeled examples and I will adapt the tooling.

**What "female" means to the model.** There is no single standard feature. A model learns a mix of cues
(clothing shape, hair, build, posture, accessories) and some people are ambiguous at a distance. That is why
the app is fail-closed: unsure means blurred, and the threshold you pick trades missed women against
over-blurred men.

## Known limits

- Appearance-based classification makes mistakes (distance, angle, clothing, lighting). It is a
  best-effort filter, not a guarantee.
- "Married woman" cannot be reliably seen, so it is not attempted.
- Android: with the CLIP fallback a man takes about a second to be cleared; with a trained classifier it takes
  a few frames. Family members are recognized as soon as their face is clearly visible. Anyone whose face
  cannot be seen stays blurred until the classifier clears them.
- The face-gender reading can be badly wrong (it called a woman in a uniform and cap male with 100%
  confidence), so it is only used to add blur, never to remove it.
- Speed depends on the phone; try Fast if it feels choppy.
- Avatar face / lip-sync overlay and the glasses version are later phases.
