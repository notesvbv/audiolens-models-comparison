# AudioLens J5 : TTS Manual MOS Evaluation

**Evaluator:** Vaibhav  
**Date:** 18 March 2026  
**Scale:** 1 = Bad &nbsp;&nbsp; 2 = Poor &nbsp;&nbsp; 3 = Fair &nbsp;&nbsp; 4 = Good &nbsp;&nbsp; 5 = Excellent

---

## Scoring Sheet

| Sent # | Category | Text (first 6 words) | Edge TTS | Kokoro |
|---|---|---|---|---|
| 01 | harvard | The birch canoe slid on... | 4 | 4 |
| 08 | harvard | Four hours of steady work... | 5 | 5 |
| 17 | harvard | The source of the big... | 5 | 5 |
| 20 | harvard | The wrist was badly strained... | 4 | 4 |
| 21 | restaurant_bill | Your total for this evening... | 5 | 4 |
| 24 | atm_receipt | Cash withdrawal of one hundred... | 5 | 5 |
| 27 | medicine_label | Paracetamol five hundred milligrams. Take... | 5 | 5 |
| 28 | medicine_label | Amoxicillin two hundred and fifty... | 4 | 5 |
| 30 | syrup_label | Benylin cough syrup. Adults and... | 5 | 5 |
| 31 | syrup_label | Paracetamol oral suspension, two hundred... | 4 | 5 |
| 33 | prescription | Patient: Mrs. Sarah Ahmed. Prescribed... | 5 | 5 |
| 34 | prescription | Doctor James Wilson prescribes Omeprazole... | 4 | 5 |
| 36 | utility_bill | Your electricity bill for March... | 5 | 5 |
| 38 | utility_bill | Water usage for the quarter... | 4 | 5 |
| 39 | menu | Today's special: pan-seared salmon with... | 5 | 5 |
| 41 | menu | Dietary note: this dish contains... | 5 | 4 |
| 42 | letter | Dear Mr. Patel, we are... | 5 | 5 |
| 45 | specification | Samsung Galaxy display: six point... | 4 | 4 |
| 48 | notice | Road closure notice: High Street... | 5 | 5 |
| 50 | notice | Strong winds forecast for coastal... | 5 | 5 |

---

## Summary

| Model | Total Score | Avg MOS |
|---|---|---|
| Edge TTS | 93 / 100 | 4.65 |
| Kokoro | 95 / 100 | **4.75** ✅ |

---

## Per-Category Averages

| Category | Edge TTS | Kokoro |
|---|---|---|
| harvard (01, 08, 17, 20) | 4.50 | 4.50 |
| restaurant_bill (21) | 5.00 | 4.00 |
| atm_receipt (24) | 5.00 | 5.00 |
| medicine_label (27, 28) | 4.50 | 5.00 |
| syrup_label (30, 31) | 4.50 | 5.00 |
| prescription (33, 34) | 4.50 | 5.00 |
| utility_bill (36, 38) | 4.50 | 5.00 |
| menu (39, 41) | 5.00 | 4.50 |
| letter (42) | 5.00 | 5.00 |
| specification (45) | 4.00 | 4.00 |
| notice (48, 50) | 5.00 | 5.00 |

---

## What We Listened For

- **Naturalness** — does it sound human or robotic?
- **Clarity** — are all words clearly pronounced?
- **Prosody** — does the rhythm and intonation feel right?
- **Medical accuracy** — are drug names and numbers correct? (critical for AudioLens)

---

## Notes / Observations
*"Both models produced high-quality, natural-sounding speech across all document categories, with mean opinion scores of 4.75 (Kokoro) and 4.65 (Edge TTS) respectively. While the quality difference is marginal, Kokoro's slight advantage on medical and prescription content — the most critical document types for AudioLens users - combined with its fully offline capability make it the preferred choice for the AudioLens TTS pipeline."*
