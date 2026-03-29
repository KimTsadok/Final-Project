# Week 3 Evaluation

## Goal
By the end of Week 3, the system should produce:
- `VideoFeatures.json`
- `VideoInterpretation.json`

for each tested video, and compare the predicted narrative phase against a human expected phase.

## Evaluation Table

## Evaluation Table

| video_id | expected_phase | predicted_phase | match | complexity_score | notes |
|----------|----------------|-----------------|-------|------------------|-------|
| ACCEDE09230 | Calm | Calm | Yes | 0.5232 | Quiet domestic living-room scene; high object diversity but low pace, so Calm is reasonable (visually rich but not dynamic) |
| ACCEDE09231 | Calm | Calm | Yes | 0.3679 | Calm scene but still has movement, thus Calm and not Static; medium object entropy |
| ACCEDE09232 | Static | Calm | No | 0.4226 | Low-motion scene; interaction density slightly above Static threshold, so classified as Calm; threshold may be too strict |

## Notes
- This evaluation is qualitative and meant as a Week 3 sanity check.
- `expected_phase` is assigned manually based on human interpretation of the clip(Calm/Dynamic/Dense/Static).
- `predicted_phase` and `complexity_score` are taken from `VideoInterpretation.json`.
- A scene can be narratively calm while still having moderate visual complexity 
(e.g., ACCEDE09230), due to high object entropy but low motion.
- Misclassification cases (e.g., ACCEDE09232) suggest that current thresholds for Static vs Calm may require tuning.
