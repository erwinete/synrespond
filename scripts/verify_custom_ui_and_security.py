#!/usr/bin/env python3
"""CI-верификация кастомного UI и безопасности synrespond (ТЗ, раздел 8.3).

Проверяет отсутствие сторонних библиотек иконок, системных Unicode-смайликов
и вызовов shell=True в переведенных на дизайн-систему каталогах.
"""

import pathlib
import re
import sys

FORBIDDEN_ICON_LIBS = (
    "lucide-svelte",
    "lucide-react",
    "@heroicons",
    "@iconify",
    "fontawesome",
    "material-icons",
    "@tabler/icons",
)

EMOJI_RANGE_RE = re.compile(
    "["
    "\U0001F300-\U0001F9FF"
    "\U0001FA00-\U0001FAFF"
    "\U00002600-\U000027BF"
    "]+",
    flags=re.UNICODE,
)


def main() -> int:
    root = pathlib.Path(__file__).resolve().parents[1]
    target_dirs = [
        root / "open-webui-fork" / "src" / "lib" / "components" / "synrespond",
        root / "open-webui-fork" / "src" / "lib" / "components" / "workspace" / "Intents",
        root / "open-webui-fork" / "src" / "lib" / "components" / "workspace" / "Knowledge" / "HippoRAGGraphViewer.svelte",
        root / "open-webui-fork" / "src" / "lib" / "components" / "admin" / "Ragas",
    ]
    errors: list[str] = []

    for d in target_dirs:
        if not d.exists():
            continue
        paths = d.rglob("*") if d.is_dir() else [d]
        for path in paths:
            if not path.is_file() or path.suffix not in (".svelte", ".ts", ".js", ".css"):
                continue
            content = path.read_text(encoding="utf-8", errors="ignore")
            for lib in FORBIDDEN_ICON_LIBS:
                if lib in content:
                    errors.append(f"[UI-LIB-VIOLATION] {path}: запрещенный импорт '{lib}'")
            for match in EMOJI_RANGE_RE.finditer(content):
                errors.append(f"[UI-EMOJI-VIOLATION] {path}: системный эмодзи '{match.group(0)}'")

    mcp_file = root / "open-webui-fork" / "backend" / "open_webui" / "utils" / "mcp_harness.py"
    if mcp_file.exists():
        mcp_code = mcp_file.read_text(encoding="utf-8", errors="ignore")
        if "shell=True" in mcp_code or "create_subprocess_shell" in mcp_code:
            errors.append(f"[SEC-01-VIOLATION] {mcp_file}: обнаружен запрещенный вызов shell=True")

    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1

    print("SYNRESPOND VERIFICATION PASSED: 0 stock icons, 0 system emojis, 0 RCE shell=True calls.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
