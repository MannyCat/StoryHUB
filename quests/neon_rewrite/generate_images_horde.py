import argparse
import base64
import csv
import hashlib
import json
import re
import time
import urllib.parse
import urllib.request
import urllib.error
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple


HORDE_BASE = "https://stablehorde.net/api/v2"


HTTP_RETRIES = 6
HTTP_BACKOFF_BASE_SEC = 2.0
HTTP_BACKOFF_MAX_SEC = 60.0


def _retry_sleep_from_error(err: urllib.error.HTTPError, attempt: int) -> float:
    retry_after = err.headers.get("Retry-After") if hasattr(err, "headers") else None
    exp = min(HTTP_BACKOFF_MAX_SEC, HTTP_BACKOFF_BASE_SEC * (2 ** max(0, attempt - 1)))
    if retry_after:
        try:
            ra = max(1.0, float(retry_after))
            # Some proxies return Retry-After=1 even when you should back off more.
            base = max(ra, exp)
            return min(HTTP_BACKOFF_MAX_SEC, base + random.uniform(0.0, 1.0))
        except Exception:
            pass
    # Exponential backoff with a soft cap + tiny jitter.
    return min(HTTP_BACKOFF_MAX_SEC, exp + random.uniform(0.0, 1.0))


def _stable_seed(text: str) -> int:
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def _http_json(method: str, url: str, *, headers: Dict[str, str], payload: Optional[dict] = None) -> dict:
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers = {**headers, "Content-Type": "application/json"}

    last_err: Optional[Exception] = None
    for attempt in range(1, HTTP_RETRIES + 2):
        try:
            req = urllib.request.Request(url, method=method, headers=headers, data=data)
            with urllib.request.urlopen(req, timeout=180) as resp:
                raw = resp.read()
            return json.loads(raw.decode("utf-8"))
        except urllib.error.HTTPError as e:
            last_err = e
            if e.code == 429 and attempt <= HTTP_RETRIES + 1:
                sleep_s = _retry_sleep_from_error(e, attempt)
                print(f"WARN: HTTP 429 for {url}. Sleep {sleep_s:.1f}s and retry ({attempt}/{HTTP_RETRIES + 1})")
                time.sleep(sleep_s)
                continue
            raise
        except Exception as e:
            last_err = e
            raise
    raise RuntimeError(f"HTTP JSON failed for {url}: {last_err}")


def _http_bytes(url: str) -> bytes:
    last_err: Optional[Exception] = None
    for attempt in range(1, HTTP_RETRIES + 2):
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "StoryHubQuestImageGen/1.0",
                },
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=180) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            last_err = e
            if e.code == 429 and attempt <= HTTP_RETRIES + 1:
                sleep_s = _retry_sleep_from_error(e, attempt)
                print(f"WARN: HTTP 429 while downloading. Sleep {sleep_s:.1f}s and retry ({attempt}/{HTTP_RETRIES + 1})")
                time.sleep(sleep_s)
                continue
            raise
        except Exception as e:
            last_err = e
            raise
    raise RuntimeError(f"HTTP download failed for {url}: {last_err}")


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


def _read_quest_assets(quest_path: Path) -> Tuple[Optional[str], Optional[str], List[Tuple[str, str]]]:
    data = json.loads(quest_path.read_text(encoding="utf-8"))
    meta = data.get("meta", {}) or {}

    cover_file = meta.get("coverImage")
    title = (meta.get("title") or "").strip()
    description = (meta.get("description") or "").strip()

    cover_prompt = None
    if cover_file:
        cover_prompt = (
            "Cover art, cyberpunk horror, neon, cinematic lighting, high detail. "
            f"Title: {title}. {description}"
        ).strip()

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

    return cover_file, cover_prompt, endings


def _queue_generation(
    *,
    apikey: str,
    prompt: str,
    width: int,
    height: int,
    steps: int,
    cfg_scale: float,
    seed: int,
    negative_prompt: str,
    n: int,
    nsfw: bool,
) -> str:
    url = f"{HORDE_BASE}/generate/async"

    payload = {
        "prompt": prompt,
        "params": {
            "width": width,
            "height": height,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "seed": str(seed),
            "n": n,
        },
        "nsfw": nsfw,
        "shared": False,
        "trusted_workers": False,
        "slow_workers": True,
        "censor_nsfw": not nsfw,
        "workers": [],
        "models": [],
    }

    # AI Horde uses "###" negative prompt separator in many clients.
    if negative_prompt.strip():
        payload["prompt"] = f"{prompt} ### {negative_prompt.strip()}"

    headers = {
        "apikey": apikey,
        "Client-Agent": "StoryHubQuest/1.0 (batch image gen)",
    }

    res = _http_json("POST", url, headers=headers, payload=payload)
    gen_id = res.get("id")
    if not gen_id:
        raise RuntimeError(f"Unexpected response from horde: {res}")
    return gen_id


def _soften_prompt(prompt: str) -> str:
    # Reduce likelihood of NSFW classifier hits while keeping the scene usable.
    repl = [
        ("nude", "clothed"),
        ("naked", "clothed"),
        ("nudity", ""),
        ("sex", ""),
        ("erotic", ""),
        ("porn", ""),
        ("rape", ""),
        ("genitals", ""),
    ]
    out = prompt
    low = out.lower()
    for a, b in repl:
        if a in low:
            out = re.sub(re.escape(a), b, out, flags=re.IGNORECASE)
            low = out.lower()
    out = re.sub(r"\s{2,}", " ", out).strip()
    return out


def _wait_done(*, gen_id: str, poll_interval: float, timeout_sec: int) -> None:
    url = f"{HORDE_BASE}/generate/check/{gen_id}"
    headers = {"Client-Agent": "StoryHubQuest/1.0 (batch image gen)"}

    started = time.time()
    while True:
        res = _http_json("GET", url, headers=headers)
        done = bool(res.get("done"))
        if done:
            return

        if time.time() - started > timeout_sec:
            raise TimeoutError(f"Timeout waiting for generation {gen_id}")

        time.sleep(max(0.5, poll_interval))


def _fetch_result(*, gen_id: str) -> dict:
    url = f"{HORDE_BASE}/generate/status/{gen_id}"
    headers = {"Client-Agent": "StoryHubQuest/1.0 (batch image gen)"}
    return _http_json("GET", url, headers=headers)


_DATA_URL_RE = re.compile(r"^data:[^;]+;base64,", re.IGNORECASE)


def _save_generation_as_jpg(*, out_path: Path, b64_webp: str, jpg_only: bool) -> None:
    b64_webp = (b64_webp or "").strip()

    # AI Horde can return either a base64 string OR a direct URL (depending on settings and backend).
    # Detect URLs early and download bytes.
    if b64_webp.startswith("http://") or b64_webp.startswith("https://"):
        try:
            raw = _http_bytes(b64_webp)
        except Exception as e:
            raise RuntimeError(f"Failed to download image URL for '{out_path.name}': {e}")
    else:
        src_prefix = b64_webp[:80]
        src_len = len(b64_webp)

        b64_webp = _DATA_URL_RE.sub("", b64_webp)

        try:
            raw = base64.b64decode(b64_webp, validate=False)
        except Exception as e:
            if jpg_only:
                raise RuntimeError(
                    f"Base64 decode failed for '{out_path.name}': {e}. src_prefix={src_prefix!r} src_len={src_len}"
                )
            print(f"WARN: base64 decode failed for '{out_path.name}': {e}")
            return

    # Try to convert webp->jpg if Pillow is installed.
    try:
        from PIL import Image  # type: ignore

        import io

        im = Image.open(io.BytesIO(raw))
        im = im.convert("RGB")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        im.save(out_path, format="JPEG", quality=92)
        return
    except Exception as e:
        if jpg_only:
            magic = raw[:16]
            try:
                bin_path = out_path.with_suffix(out_path.suffix + ".bin")
                bin_path.write_bytes(raw)
            except Exception:
                pass
            raise RuntimeError(
                f"JPG conversion failed for '{out_path.name}'. Magic bytes={magic!r} len={len(raw)}. Original error: {e}"
            )
        # Fallback: save as .webp next to requested file.
        print(
            f"WARN: JPG conversion failed for '{out_path.name}', saved WEBP instead. Error: {e}"
        )
        # Always dump the raw bytes for troubleshooting.
        try:
            bin_path = out_path.with_suffix(out_path.suffix + ".bin")
            bin_path.write_bytes(raw)
            print(
                f"DEBUG: wrote raw bytes to '{bin_path.name}'. Magic={raw[:16]!r} len={len(raw)}"
            )
        except Exception as dump_e:
            print(f"WARN: failed to write raw dump for '{out_path.name}': {dump_e}")
        webp_path = out_path.with_suffix(".webp")
        webp_path.parent.mkdir(parents=True, exist_ok=True)
        webp_path.write_bytes(raw)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate quest images using AI Horde (StableHorde) and save into quest folder."
    )
    parser.add_argument(
        "--quest-dir",
        default=str(Path(__file__).resolve().parent),
        help="Quest directory (default: script folder)",
    )
    parser.add_argument(
        "--apikey",
        default="0000000000",
        help="AI Horde API key. Use 0000000000 for anonymous.",
    )
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--cfg", type=float, default=7.0)
    parser.add_argument(
        "--negative",
        default=(
            "worst quality, low quality, blurry, watermark, text, logo, signature, extra fingers, deformed"
        ),
    )
    parser.add_argument("--sleep", type=float, default=1.0, help="Delay between jobs")
    parser.add_argument("--poll", type=float, default=3.0, help="Polling interval for status")
    parser.add_argument("--timeout", type=int, default=900, help="Per-image timeout in seconds")
    parser.add_argument(
        "--http-retries",
        type=int,
        default=6,
        help="How many retries to do on HTTP 429 (rate limit) for API/download requests (default: 6).",
    )
    parser.add_argument(
        "--http-backoff",
        type=float,
        default=2.0,
        help="Base seconds for exponential backoff on HTTP 429 (default: 2.0).",
    )
    parser.add_argument(
        "--http-backoff-max",
        type=float,
        default=60.0,
        help="Max seconds to sleep between HTTP 429 retries (default: 60.0).",
    )
    parser.add_argument("--max", type=int, default=0, help="Limit number of images (0=all)")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--nsfw", action="store_true")
    parser.add_argument(
        "--jpg-only",
        action="store_true",
        help="Fail if JPG conversion fails (do not write .webp fallback).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print extra diagnostics (img prefix/length, generation keys).",
    )
    parser.add_argument("--include-cover", action="store_true")
    parser.add_argument("--include-endings", action="store_true")
    parser.add_argument(
        "--retry-censored",
        type=int,
        default=1,
        help="How many times to retry when Horde returns a censored placeholder (default: 1).",
    )
    parser.add_argument(
        "--soften-on-censored",
        action="store_true",
        help="On censored result, retry with a softened prompt and stronger negative prompt.",
    )

    args = parser.parse_args()

    # urllib encodes HTTP headers as latin-1. If the apikey contains non-latin characters
    # (e.g. Cyrillic), requests will fail with a confusing codec error.
    try:
        (args.apikey or "").encode("latin-1")
    except UnicodeEncodeError:
        raise SystemExit(
            "Invalid --apikey: it must contain only latin characters (ASCII/latin-1). "
            "Please paste the exact AI Horde API key (usually looks like random letters/numbers)."
        )

    global HTTP_RETRIES, HTTP_BACKOFF_BASE_SEC, HTTP_BACKOFF_MAX_SEC
    HTTP_RETRIES = max(0, int(args.http_retries))
    HTTP_BACKOFF_BASE_SEC = max(0.1, float(args.http_backoff))
    HTTP_BACKOFF_MAX_SEC = max(1.0, float(args.http_backoff_max))

    quest_dir = Path(args.quest_dir)
    csv_path = quest_dir / "prompts_manifest.csv"
    quest_path = quest_dir / "quest.json"

    if not csv_path.exists():
        raise SystemExit(f"Missing {csv_path}. Run generate_prompts_manifest.py first.")
    if not quest_path.exists():
        raise SystemExit(f"Missing {quest_path}.")

    scenes = _read_scene_prompts(csv_path)

    cover_file, cover_prompt, endings = _read_quest_assets(quest_path)

    jobs: List[Tuple[str, str]] = [(r["file"], r["prompt"]) for r in scenes]

    if args.include_cover and cover_file and cover_prompt:
        jobs.insert(0, (cover_file, cover_prompt))

    if args.include_endings:
        for fname, p in endings:
            jobs.append((fname, p))

    if args.max and args.max > 0:
        jobs = jobs[: args.max]

    ok = 0
    skipped = 0
    failed = 0

    for idx, (fname, prompt) in enumerate(jobs, start=1):
        out_path = quest_dir / fname

        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue

        seed = _stable_seed(fname)
        base_prompt = (
            "Cyberpunk horror, neon, cinematic lighting, high detail. " + prompt
        ).strip()

        try:
            attempt = 0
            retries = max(0, int(args.retry_censored))
            prompt_for_attempt = base_prompt
            negative_for_attempt = args.negative

            while True:
                attempt += 1
                print(f"[{idx}/{len(jobs)}] QUEUE -> {fname}")
                gen_id = _queue_generation(
                    apikey=args.apikey,
                    prompt=prompt_for_attempt,
                    width=args.width,
                    height=args.height,
                    steps=args.steps,
                    cfg_scale=args.cfg,
                    seed=seed,
                    negative_prompt=negative_for_attempt,
                    n=1,
                    nsfw=args.nsfw,
                )

                print(f"[{idx}/{len(jobs)}] WAIT  -> {gen_id}")
                _wait_done(gen_id=gen_id, poll_interval=args.poll, timeout_sec=args.timeout)

                status = _fetch_result(gen_id=gen_id)
                gens = status.get("generations") or []
                if not gens:
                    raise RuntimeError(f"No generations returned for {gen_id}: {status}")

                g0 = gens[0]
                if bool(args.debug):
                    try:
                        keys = sorted(list(g0.keys())) if isinstance(g0, dict) else []
                        img_val = g0.get("img") if isinstance(g0, dict) else None
                        img_str = str(img_val) if img_val is not None else ""
                        cens = bool(g0.get("censored")) if isinstance(g0, dict) else False
                        print(
                            f"DEBUG: generation keys={keys} censored={cens} img_prefix={img_str[:80]!r} img_len={len(img_str)}"
                        )
                    except Exception as dbg_e:
                        print(f"DEBUG: failed to print generation debug info: {dbg_e}")

                censored = bool(g0.get("censored")) if isinstance(g0, dict) else False
                if censored:
                    # Horde returned a placeholder because we asked to block NSFW.
                    msg = (
                        "CENSORED result returned by AI Horde (NSFW detected and client requested blocking)."
                    )
                    if not args.nsfw:
                        msg += " Consider re-running with --nsfw to allow NSFW (may still be filtered by some workers)."
                    print(f"[{idx}/{len(jobs)}] WARN  {fname}: {msg}")

                    if attempt <= retries:
                        if bool(args.soften_on_censored):
                            prompt_for_attempt = _soften_prompt(base_prompt)
                            negative_for_attempt = (
                                (negative_for_attempt or "")
                                + ", nudity, naked, explicit, porn, sex, erotic"
                            ).strip(", ")
                            print(
                                f"[{idx}/{len(jobs)}] RETRY {fname}: softened prompt + stronger negative (attempt {attempt}/{retries + 1})"
                            )
                        else:
                            print(
                                f"[{idx}/{len(jobs)}] RETRY {fname}: censored (attempt {attempt}/{retries + 1})"
                            )
                        continue

                    raise RuntimeError(msg)

                img_b64 = g0.get("img") if isinstance(g0, dict) else None
                if not img_b64:
                    raise RuntimeError(f"No img field in generation for {gen_id}: {g0}")

                _save_generation_as_jpg(
                    out_path=out_path,
                    b64_webp=img_b64,
                    jpg_only=bool(args.jpg_only),
                )
                ok += 1
                print(f"[{idx}/{len(jobs)}] OK    -> {fname}")
                break

        except Exception as e:
            failed += 1
            print(f"[{idx}/{len(jobs)}] FAILED {fname}: {e}")

        time.sleep(max(0.0, float(args.sleep)))

    print(f"DONE. ok={ok} skipped={skipped} failed={failed} out={quest_dir}")
    print(
        "NOTE: AI Horde often returns WEBP. If you don't have Pillow installed, images are saved as .webp next to the .jpg name."
    )


if __name__ == "__main__":
    main()
