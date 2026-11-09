"""Record a successfully compiled debug APK without claiming device execution.

Run after ``npx cap sync`` and ``android/gradlew assembleDebug``:
  ANDROID_HOME=/path/to/sdk JAVA_HOME=/path/to/jdk python3 scripts/native-receipt.py
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SDK = Path(os.environ["ANDROID_HOME"])
JAVA = Path(os.environ["JAVA_HOME"]) / "bin/java"
APK = ROOT / "android/app/build/outputs/apk/debug/app-debug.apk"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def command(arguments: list[str]) -> str:
    return subprocess.check_output(arguments, stderr=subprocess.STDOUT, text=True).strip()


def revision(folder: str) -> str:
    properties = (SDK / folder / "source.properties").read_text()
    return re.search(r"^Pkg.Revision=(.+)$", properties, re.MULTILINE).group(1)


signature = command([
    str(SDK / "build-tools/35.0.0/apksigner"), "verify", "--verbose", str(APK)
])
with zipfile.ZipFile(APK) as archive:
    # Native packaging must contain precisely the current production web build.
    for file in sorted((ROOT / "dist").rglob("*")):
        if file.is_file():
            name = "assets/public/" + file.relative_to(ROOT / "dist").as_posix()
            assert archive.read(name) == file.read_bytes(), f"APK has stale assets: {name}"
    config = json.loads(archive.read("assets/capacitor.config.json"))
    assert "server" not in config, "Release artifact points at a development server"
    assert config["appId"] == "com.josesanchez.agentfield"

receipt = {
    "schema_version": 1,
    "recorded_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "host_os": platform.system(),
    "app_id": config["appId"],
    "version": json.loads((ROOT / "package.json").read_text())["version"],
    "android": {
        "build_command": "./gradlew --no-daemon assembleDebug",
        "artifact": "android/app/build/outputs/apk/debug/app-debug.apk",
        "size_bytes": APK.stat().st_size,
        "sha256": sha256(APK),
        "compile_sdk": 35,
        "min_sdk": 23,
        "target_sdk": 35,
        "build_tools": revision("build-tools/35.0.0"),
        "command_line_tools": revision("cmdline-tools/latest"),
        "platform_tools": revision("platform-tools"),
        "gradle": "8.11.1",
        "android_gradle_plugin": "8.7.2",
        "java": command([str(JAVA), "-version"]).splitlines()[0],
        "signature_verification": signature.splitlines(),
        "web_assets_byte_identical_to_dist": True,
        "native_runtime_tested": False,
        "scope": "Compiled and signed with a local debug key; no emulator or physical-device execution.",
    },
    "ios": {
        "project": "ios/App/App.xcodeproj",
        "dependency_manager": "Swift Package Manager",
        "minimum_ios": "14.0",
        "build_verified_locally": False,
        "native_runtime_tested": False,
        "scope": "Native source synced on Linux. Unsigned simulator build configured in mobile-app.yml on macOS; no local Xcode or App Store signing.",
    },
    "capacitor_version": json.loads((ROOT / "package.json").read_text())["dependencies"]["@capacitor/core"],
    "source_sha256": {
        name: sha256(ROOT / name) for name in (
            "package-lock.json", "capacitor.config.ts",
            "android/gradle/wrapper/gradle-wrapper.properties",
            "android/app/build.gradle", "android/app/src/main/AndroidManifest.xml",
            "android/app/src/main/res/values/styles.xml",
            "ios/App/App.xcodeproj/project.pbxproj", "ios/App/CapApp-SPM/Package.swift",
        )
    },
    "production_web_sha256": {
        file.relative_to(ROOT / "dist").as_posix(): sha256(file)
        for file in sorted((ROOT / "dist").rglob("*")) if file.is_file()
    },
}
download_receipt = SDK / "toolchain-download.json"
if download_receipt.is_file():
    receipt["android"]["sdk_download"] = json.loads(download_receipt.read_text())
output = ROOT / "demo/native-validation.json"
output.write_text(json.dumps(receipt, indent=2) + "\n")
print(f"Recorded {output}; APK {receipt['android']['sha256']}")
