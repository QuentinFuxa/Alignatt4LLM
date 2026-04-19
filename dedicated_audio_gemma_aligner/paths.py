from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parent
ARTIFACT_DIR = MODULE_DIR / "artifacts" / "feature_aligner"
SMALL_SPLIT_MANIFEST = ARTIFACT_DIR / "split_manifest.json"
FULL_SPLIT_MANIFEST = ARTIFACT_DIR / "split_manifest_full.json"
TEACHER_DIR = ARTIFACT_DIR / "teachers"
ACL_GOLD_DIR = PROJECT_ROOT / "acl-speech" / "segmented_wavs" / "gold"
ACL_TEXT_FILE = PROJECT_ROOT / "acl-speech" / "text" / "txt" / "ACL.6060.eval.en-xx.en.txt"
