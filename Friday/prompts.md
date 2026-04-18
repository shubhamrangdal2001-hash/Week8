# AI Prompts Used — Week 08 · Friday

**Policy:** Every AI-assisted sub-step includes the exact prompt used and a critique.

---

## Prompt Used (All Sub-steps)

```
You are an expert AI/ML engineer specializing in medical imaging and transfer learning.
I am working on a chest X-ray classification problem using a small dataset (~520 labeled 
samples across 5 conditions + 30 unlabeled images). The goal is to design a clinically 
safe, explainable model using transfer learning.

Follow these strict requirements step-by-step:

1. DATA UNDERSTANDING
   - Analyse label distribution and identify class imbalance.
   - Highlight risks of imbalance in a clinical setting (false negatives especially).
   - Check for subgroup bias (hospital source, image quality).
   - Provide clear insights that will influence modeling decisions.

2. MODEL SELECTION + FEATURE EXTRACTION
   - Select a suitable pre-trained CNN (e.g., ResNet, EfficientNet).
   - Justify selection based on dataset size and generalization.
   - Implement feature extraction (freeze backbone, train classifier head).
   - Provide evaluation metrics: per-class precision, recall, F1 + confusion matrix.
   - Identify weakest classes and explain clinical implications.

3. FINE-TUNING STRATEGY
   - Unfreeze top layers and fine-tune with a low learning rate.
   - Compare with feature extraction using per-class recall (critical for medical risk).
   - Clearly conclude which approach is safer for deployment and why.

4. MODEL EXPLAINABILITY
   - Generate Grad-CAM/saliency maps.
   - Compare correct predictions vs misclassifications.
   - Explain what the model focuses on.
   - Summarize findings in 2 simple sentences for a radiologist.

5. ME1 PREPARATION (SELF-SYNTHESIS)
   - Explain one weak topic (from ML/DL concepts) in ~200 words.
   - Create 2 interview questions with answers.
   - Ensure clarity and conceptual depth.

6. UNLABELED DATA PREDICTION
   - Predict labels for 30 unlabeled images.
   - Provide confidence scores.
   - Ensure outputs are structured and interpretable.

7. (OPTIONAL – ADVANCED)
   - Compare: Feature extraction, Fine-tuning, Training from scratch.
   - Use identical evaluation setup.
   - Conclude viability of training from scratch.

8. TRIAGE SYSTEM DESIGN
   - Create 3-tier decision system: auto-classify, human review, reject/rescan.
   - Define confidence thresholds with justification.
   - Estimate false-negative risk.

IMPORTANT CONSTRAINTS:
- Focus on clinical safety (false negatives are most costly).
- Avoid generic explanations — use reasoning and evidence.
- Code should be modular, production-quality (functions, no hardcoding).
- Include comments explaining design choices.
- PyTorch preferred.
```

---

## Critique

### ✅ What the AI Output Got Right

1. **EfficientNet-B0 selection** — Correctly identified that 5.3M parameters is appropriate for n=520; would have overfitted with ResNet-50 (25M params).
2. **FocalLoss rationale** — Correctly linked class imbalance to focal loss and explained the γ parameter's role in focusing on hard examples.
3. **Grad-CAM implementation** — Hook-based implementation is architecturally correct; gradient pooling → weighted sum → ReLU is faithful to Selvaraju et al. 2017.
4. **Triage design** — Correctly framed thresholds around clinical cost structure (FN > FP in screening).
5. **Discriminative learning rates** — Correctly suggested lower LR for backbone than for head during fine-tuning.

### ✏️ What Was Modified and Why

| Modification | Reason |
|---|---|
| Added `set_seed()` globally | AI omitted it; without this, results differ each run — makes reproducibility impossible |
| Converted all config to `Config` dataclass | AI had several hardcoded values (e.g., `lr=0.001`, `epochs=20`); production code must have named constants |
| Added `generate_synthetic_metadata()` fallback | AI assumed CSV was present; notebook must run in clean TA environment without LMS access |
| Added `try/except` around all model file I/O | Bare `torch.load()` crashes without helpful messages; wrapped for production safety |
| Changed primary metric display from accuracy to recall | AI kept showing accuracy first; in clinical setting recall must be the headline metric |
| Added `apply_clinical_override()` for dangerous classes | AI's triage sent high-confidence COVID/TB to auto-classify; this is clinically unacceptable — always needs human review |
| Added hospital-stratified heatmap in Sub-step 1 | AI only checked overall class counts; subgroup bias by hospital is equally important |
| Added `estimate_false_negative_risk()` with calibration caveat | AI computed FN count but did not note that small-dataset models are typically overconfident, inflating FN risk |

### ⚠️ Limitations and Risks Not Fully Addressed by AI

1. **Calibration**: AI did not mention that model confidence at n=520 is likely poorly calibrated. Temperature scaling (or Platt scaling) is required before trusting triage thresholds in deployment.

2. **No actual DICOM/pixel loading**: The AI-generated code assumed image arrays would be available. This notebook uses synthetic tensors as a faithful standin; real deployment requires DICOM parsing (pydicom) and CLAHE preprocessing.

3. **Data leakage risk**: AI's initial version did not enforce stratified splits — a random split on n=520 could leave zero examples of a rare class in the test set, making evaluation impossible for that class.

4. **Batch Normalisation in frozen layers**: AI did not address the BN-in-frozen-backbone bug (running stats drift). Covered in ME1 section but should also be patched in `build_fine_tuning_model()` by switching frozen BN layers to `.eval()` explicitly.
