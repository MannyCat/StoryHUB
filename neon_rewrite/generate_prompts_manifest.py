import csv
import json
from pathlib import Path


def main() -> None:
    quest_path = Path(__file__).with_name("quest.json")
    out_csv = Path(__file__).with_name("prompts_manifest.csv")
    out_json = Path(__file__).with_name("prompts_manifest.json")

    data = json.loads(quest_path.read_text(encoding="utf-8"))

    scenes = data.get("scenes", []) or []

    scene_rows = []
    for s in scenes:
        prompt = (s.get("imagePrompt") or "").strip()
        if not prompt:
            continue

        sid = (s.get("id") or "").strip()
        title = (s.get("title") or "").strip()

        scene_rows.append(
            {
                "sceneId": sid,
                "title": title,
                "imagePrompt": prompt,
                "suggestedFile": f"{sid}.jpg" if sid else "",
            }
        )

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["sceneId", "title", "imagePrompt", "suggestedFile"],
        )
        w.writeheader()
        for r in scene_rows:
            w.writerow(r)

    meta = data.get("meta", {}) or {}
    cover = meta.get("coverImage")

    endings = []
    for s in scenes:
        from_scene_id = s.get("id")
        for c in (s.get("choices") or []):
            if not isinstance(c, dict):
                continue
            e = c.get("ending")
            if not isinstance(e, dict):
                continue
            endings.append(
                {
                    "fromSceneId": from_scene_id,
                    "choiceId": c.get("id"),
                    "endingTitle": e.get("title"),
                    "endingImage": e.get("image"),
                }
            )

    out = {
        "coverImage": cover,
        "endingImages": endings,
        "sceneImagePrompts": scene_rows,
    }

    out_json.write_text(
        json.dumps(out, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(
        f"OK: {len(scene_rows)} scene prompts, {len(endings)} endings, cover={cover}"
    )


if __name__ == "__main__":
    main()
