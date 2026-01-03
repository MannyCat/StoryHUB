import argparse
import csv
import hashlib
import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


def _stable_seed(text: str) -> int:
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def _build_url(
    base_url: str,
    prompt: str,
    *,
    model: str,
    width: int,
    height: int,
    seed: int,
    negative_prompt: str,
    nologo: bool,
    nofeed: bool,
    safe: bool,
    enhance: bool,
    quality: str,
) -> str:
    prompt_path = urllib.parse.quote(prompt, safe="")
    query = {
        "model": model,
        "width": str(width),
        "height": str(height),
        "seed": str(seed),
        "negative_prompt": negative_prompt,
        "quality": quality,
        "enhance": "true" if enhance else "false",
        "nologo": "true" if nologo else "false",
        "nofeed": "true" if nofeed else "false",
        "safe": "true" if safe else "false",
    }
    return f"{base_url.rstrip('/')}/{prompt_path}?{urllib.parse.urlencode(query)}"


def _download(url: str, out_path: Path, timeout: int = 120) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "StoryHubQuestImageGen/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
    out_path.write_bytes(data)


def _read_scene_prompts(csv_path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            scene_id = (r.get("sceneId") or "").strip()
            prompt = (r.get("imagePrompt") or "").strip()
            suggested = (r.get("suggestedFile") or "").strip()
            if not scene_id or not prompt or not suggested:
                continue
            rows.append({"sceneId": scene_id, "prompt": prompt, "file": suggested})
    return rows


def _read_quest_assets(quest_path: Path) -> Tuple[Optional[str], List[Tuple[str, str]]]:
    data = json.loads(quest_path.read_text(encoding="utf-8"))
    meta = data.get("meta", {}) or {}

    cover_image = meta.get("coverImage")
    title = (meta.get("title") or "").strip()
    description = (meta.get("description") or "").strip()

    endings: List[Tuple[str, str]] = []
    for s in data.get("scenes", []) or []:
        for c in (s.get("choices") or []):
            if not isinstance(c, dict):
                continue
            e = c.get("ending")
            if not isinstance(e, dict):
                continue
            img = (e.get("image") or "").strip()
            if not img:
                continue
            e_title = (e.get("title") or "").strip()
            e_text = (e.get("text") or "").strip()
            endings.append((img, f"{e_title}. {e_text}".strip()))

    cover_prompt = None
    if cover_image:
        cover_prompt = (
            f"Cover art, cyberpunk horror, neon, cinematic, high detail. "
            f"Title: {title}. {description}"
        ).strip()

    # Return (cover_prompt, endings)
    # endings is list of (filename, prompt_text)
    return cover_prompt, endings


def _iter_jobs(
    scenes: Iterable[Dict[str, str]],
    *,
    include_cover: bool,
    include_endings: bool,
    cover_prompt: Optional[str],
    cover_filename: Optional[str],
    endings: List[Tuple[str, str]],
) -> List[Tuple[str, str]]:
    jobs: List[Tuple[str, str]] = []

    for r in scenes:
        jobs.append((r["file"], r["prompt"]))

    if include_cover and cover_prompt and cover_filename:
        jobs.insert(0, (cover_filename, cover_prompt))

    if include_endings:
        for fname, p in endings:
            jobs.append((fname, p))

    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate quest images using Pollinations (no-key public endpoint) and save into quest folder."
        )
    )
    parser.add_argument(
        "--quest-dir",
        default=str(Path(__file__).resolve().parent),
        help="Quest directory (default: script folder)",
    )
    parser.add_argument(
        "--base-url",
        default="https://image.pollinations.ai/prompt",
        help="Pollinations image endpoint base URL",
    )
    parser.add_argument("--model", default="flux", help="Model name (e.g., flux, turbo)")
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--quality", default="medium", choices=["low", "medium", "high", "hd"])
    parser.add_argument("--enhance", action="store_true", help="Let AI enhance prompts")
    parser.add_argument("--safe", action="store_true", help="Enable safety filters")
    parser.add_argument("--nologo", action="store_true", help="Try to remove watermark")
    parser.add_argument("--nofeed", action="store_true", help="Try to avoid public feed")
    parser.add_argument(
        "--negative",
        default=(
            "worst quality, low quality, blurry, watermark, text, logo, signature, extra fingers, deformed"
        ),
        help="Negative prompt",
    )
    parser.add_argument("--sleep", type=float, default=1.2, help="Delay between requests")
    parser.add_argument("--max", type=int, default=0, help="Limit number of images (0=all)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    parser.add_argument(
        "--include-cover",
        action="store_true",
        help="Also generate cover.jpg using meta title/description",
    )
    parser.add_argument(
        "--include-endings",
        action="store_true",
        help="Also generate ending images using ending title/text",
    )

    args = parser.parse_args()

    quest_dir = Path(args.quest_dir)
    csv_path = quest_dir / "prompts_manifest.csv"
    quest_path = quest_dir / "quest.json"

    if not csv_path.exists():
        raise SystemExit(f"Missing {csv_path}. Run generate_prompts_manifest.py first.")
    if not quest_path.exists():
        raise SystemExit(f"Missing {quest_path}.")

    scenes = _read_scene_prompts(csv_path)

    cover_prompt, endings = _read_quest_assets(quest_path)
    cover_filename = None
    try:
        meta = json.loads(quest_path.read_text(encoding="utf-8")).get("meta", {}) or {}
        cover_filename = meta.get("coverImage")
    except Exception:
        cover_filename = None

    jobs = _iter_jobs(
        scenes,
        include_cover=args.include_cover,
        include_endings=args.include_endings,
        cover_prompt=cover_prompt,
        cover_filename=cover_filename,
        endings=endings,
    )

    out_dir = quest_dir

    total = len(jobs)
    if args.max and args.max > 0:
        jobs = jobs[: args.max]

    ok = 0
    skipped = 0
    failed = 0

    for idx, (fname, prompt) in enumerate(jobs, start=1):
        out_path = out_dir / fname

        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue

        seed = _stable_seed(fname)
        full_prompt = (
            "Cyberpunk horror, neon, cinematic lighting, high detail, ultra sharp. " + prompt
        ).strip()

        url = _build_url(
            args.base_url,
            full_prompt,
            model=args.model,
            width=args.width,
            height=args.height,
            seed=seed,
            negative_prompt=args.negative,
            nologo=args.nologo,
            nofeed=args.nofeed,
            safe=args.safe,
            enhance=args.enhance,
            quality=args.quality,
        )

        try:
            print(f"[{idx}/{len(jobs)}] -> {fname}")
            _download(url, out_path)
            ok += 1
        except Exception as e:
            failed += 1
            print(f"FAILED {fname}: {e}")

        time.sleep(max(0.0, float(args.sleep)))

    print(
        f"DONE. total_jobs={total} processed={len(jobs)} ok={ok} skipped={skipped} failed={failed} out={out_dir}"
    )


if __name__ == "__main__":
    main()
