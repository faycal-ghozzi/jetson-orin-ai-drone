import os

ROOT = "/project/ai-drone-ws/src"
OUTPUT_FILE = os.path.join(ROOT, "collected_sources.txt")

def redact_env_line(line: str) -> str:
    """Replace values in .env lines with fake values, keeping only the keys."""
    if "=" in line and not line.strip().startswith("#"):
        key = line.split("=", 1)[0].strip()
        return f"{key}=REDACTED\n"
    return line

def read_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            if os.path.basename(path) == ".env":
                return "".join(redact_env_line(line) for line in f)
            return f.read()
    except Exception as e:
        return f"[!] Could not read {path}: {e}\n"

def main():
    collected_files = []

    # Find all .py and .xml files under ROOT
    for dirpath, _, filenames in os.walk(ROOT):
        for filename in filenames:
            if filename.endswith((".py", ".xml")) or filename == ".env":
                collected_files.append(os.path.join(dirpath, filename))

    collected_files.sort()  # for consistent ordering

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        for path in collected_files:
            rel_path = os.path.relpath(path, ROOT)
            out.write(f"\n{'='*80}\n")
            out.write(f"# File: {rel_path}\n")
            out.write(f"{'='*80}\n\n")
            out.write(read_file(path))
            out.write("\n\n")

    print(f"✅ Collected {len(collected_files)} files into {OUTPUT_FILE}")

if __name__ == "__main__":
    main()

