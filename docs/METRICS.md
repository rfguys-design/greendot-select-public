# Metrics — what "reliably" means

Written in Week 1. Do not change these definitions mid-season; if you must,
record the date and rerun the whole dataset so the numbers stay comparable.

## The definition of a successful capture

A capture **succeeds** when all three hold:

1. Exactly two real markers are found.
2. The crop contains the entire intended region and no adjacent content.
3. The delivered result matches the ground truth for the requested task.

## The four metrics

| Owner | Metric | Definition |
|---|---|---|
| Harper | Detection rate | captures where exactly 2 real dots are found ÷ total |
| James | Enclosure rate | crops fully containing the target with no intrusion ÷ crops attempted |
| Joanna | Routing accuracy | captures where the chosen task matched the intent ÷ total |
| Markus | End-to-end success | fully correct results ÷ total, plus median latency |

## Failure taxonomy

Every failure is classified, because the breakdown is what tells you what to fix
next. The headline number alone does not.

- `no_dots` — fewer than two markers found
- `extra_dots` — more than two green objects in frame
- `crop_truncated` — crop cut off part of the target
- `crop_intrusion` — crop included adjacent content
- `ocr_error` — text extracted incorrectly
- `routing_error` — wrong task chosen
- `api_error` — network or service failure
- `latency` — correct but slower than the target

## Dataset

`dataset/` holds labeled captures covering: daylight, warm indoor, fluorescent,
and dim light; camera angles from square to about 45°; printed text, handwriting,
math, and figures; tight and loose dot placement; plus adversarial frames
containing green highlighter or other green objects.

Target: 40+ captures by the end of Week 2, growing through the season.
