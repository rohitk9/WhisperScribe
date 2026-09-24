"""Build the Windows app folder and a release archive.

    .venv\\Scripts\\python packaging\\build.py

Output: dist/WhisperScribe/WhisperScribe.exe and release/WhisperScribe-<version>-win64.7z[.001, .002, …]
GitHub release assets must be < 2 GiB, so the archive is split into 1.9 GB volumes when needed (needs 7-Zip).
"""
import os
import shutil
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VERSION = sys.argv[1] if len(sys.argv) > 1 else "dev"


def folder_size(path):
    return sum(os.path.getsize(os.path.join(d, f)) for d, _, files in os.walk(path) for f in files)


def find_7zip():
    for candidate in (shutil.which("7z"), r"C:\Program Files\7-Zip\7z.exe", r"C:\Program Files (x86)\7-Zip\7z.exe"):
        if candidate and os.path.exists(candidate):
            return candidate
    return None


def main():
    subprocess.run([sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean",
                    "--distpath", os.path.join(ROOT, "dist"), "--workpath", os.path.join(ROOT, "build"),
                    os.path.join(ROOT, "packaging", "WhisperScribe.spec")], check=True, cwd=ROOT)
    app_dir = os.path.join(ROOT, "dist", "WhisperScribe")
    print(f"\nApp folder: {app_dir}  ({folder_size(app_dir) / 1e9:.2f} GB)")

    seven = find_7zip()
    if not seven:
        print("7-Zip not found — skipping the release archive.")
        return
    out_dir = os.path.join(ROOT, "release")
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir)
    archive = os.path.join(out_dir, f"WhisperScribe-{VERSION}-win64.7z")
    subprocess.run([seven, "a", "-t7z", "-mx=7", "-mmt=on", "-v1900m", archive, app_dir], check=True)
    for f in sorted(os.listdir(out_dir)):
        print(f"  {f}  {os.path.getsize(os.path.join(out_dir, f)) / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
