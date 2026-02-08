import argparse
import json
import mimetypes
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def extract_image_urls(issue_body: str) -> list[str]:
    markdown_urls = re.findall(r"!\[[^\]]*]\((https?://[^)\s]+)\)", issue_body, flags=re.IGNORECASE)
    html_urls = re.findall(r'<img[^>]+src=["\'](https?://[^"\']+)["\']', issue_body, flags=re.IGNORECASE)
    urls = []
    seen = set()
    for url in markdown_urls + html_urls:
        if url not in seen:
            seen.add(url)
            urls.append(url)
    return urls


def infer_extension(url: str, content_type: str | None) -> str:
    parsed = urlparse(url)
    suffix = Path(parsed.path).suffix.lower()
    if suffix in IMAGE_EXTS:
        return suffix
    if content_type:
        guessed = mimetypes.guess_extension(content_type.split(";")[0].strip())
        if guessed and guessed.lower() in IMAGE_EXTS:
            return guessed.lower()
    return ".png"


def download_image(url: str, save_path: Path) -> None:
    request = Request(url, headers={"User-Agent": "issue-image-worker"})
    with urlopen(request, timeout=60) as response:
        data = response.read()
        if not data:
            raise RuntimeError(f"Empty response: {url}")
        save_path.write_bytes(data)


def run_main(input_path: Path, output_path: Path) -> tuple[bool, str]:
    command = [sys.executable, "main.py", str(input_path), str(output_path)]
    result = subprocess.run(command, capture_output=True, text=True)
    ok = result.returncode == 0 and output_path.exists()
    message = (result.stdout or "") + (result.stderr or "")
    return ok, message.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--issue-body-file", required=True)
    parser.add_argument("--issue-number", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--urls-json", default="")
    args = parser.parse_args()

    issue_body = Path(args.issue_body_file).read_text(encoding="utf-8")
    urls = json.loads(args.urls_json) if args.urls_json else extract_image_urls(issue_body)

    output_root = Path(args.output_dir)
    inputs_dir = output_root / "inputs"
    results_dir = output_root / "results"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "issue_number": args.issue_number,
        "total_images": len(urls),
        "processed_images": 0,
        "items": [],
    }

    for index, url in enumerate(urls, start=1):
        item = {"index": index, "url": url, "status": "skipped"}
        try:
            ext = infer_extension(url, None)
            input_path = inputs_dir / f"input_{index}{ext}"
            download_image(url, input_path)

            output_path = results_dir / f"output_{index}.png"
            ok, message = run_main(input_path, output_path)
            item["input"] = str(input_path)
            item["output"] = str(output_path)
            item["status"] = "ok" if ok else "failed"
            item["log"] = message
            if ok:
                manifest["processed_images"] += 1
        except Exception as exc:
            item["status"] = "failed"
            item["log"] = str(exc)
        manifest["items"].append(item)

    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
