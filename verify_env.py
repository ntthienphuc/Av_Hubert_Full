import importlib
import shutil
import sys

modules = ["torch", "cv2", "mediapipe", "sentencepiece", "fairseq", "avhubert"]
failed = []
for name in modules:
    try:
        module = importlib.import_module(name)
        print(f"OK {name}: {getattr(module, '__version__', 'imported')}")
    except Exception as exc:
        failed.append((name, str(exc)))
print(f"Python: {sys.version}")
print(f"ffmpeg: {shutil.which('ffmpeg') or 'NOT FOUND (required for robust video handling)'}")
if failed:
    raise SystemExit("Import failures: " + repr(failed))
