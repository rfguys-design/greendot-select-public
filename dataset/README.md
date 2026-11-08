# Dataset

Labeled captures for `evaluate.py`. Built in Week 2 by the whole team.

## Layout

```
dataset/
  captures/
    0001.jpg          the raw frame
    0001.json         the label
```

## Label format

```json
{
  "id": "0001",
  "lighting": "daylight | warm_indoor | fluorescent | dim",
  "angle_deg": 15,
  "content": "printed | handwriting | math | figure",
  "dot_placement": "tight | loose",
  "adversarial": null,
  "intended_region": {"x": 410, "y": 292, "w": 450, "h": 130},
  "ground_truth_text": "...",
  "intended_task": "read_aloud",
  "transcript": "read this out loud"
}
```

`intended_region` is the rectangle a human would have wanted — the target for
the enclosure metric. Use `tools/label_region.py` (James, Week 2) to draw it.

## Coverage targets

At least 25 captures, including:
- all four lighting conditions
- angles from square through ~45°
- all four content types
- 4+ adversarial frames (green highlighter, green book, green pen in view)
