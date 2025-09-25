# 📊 Evaluation Plan and Results

## ✅ Metrics
- 🔎 Search suggestions: precision@k (k=5), recall@k, MRR
- 🏷️ Tag quality: human-rated relevance (1–5), coverage %, dedup rate
- 🧮 Classification: accuracy/F1 vs labeled subset, top-2 accuracy
- ⚡ Latency: P50/P90 end-to-end processing time

## 🧪 Test Scenarios
- Queries: seasonal ("winter boots"), materials ("cotton tshirt"), occasions ("formal shirt")
- Data types: apparel, footwear, accessories
- Edge cases: missing fields, short/long descriptions, mixed language, typos
- Security: input sanitization for XSS/SQL-like input

## 🔬 Methodology
- Create a 300–500 item labeled subset from `data/cleaned_product_data.csv`
- Define ground truth for category and 3–5 ideal tags per item
- Run batch inference and compute metrics with repeatable scripts
- Use blind human raters (2+) for tag relevance; average scores

## 📈 Current Results (sample placeholders)
- precision@5: TBD after labeling
- Tag relevance: TBD (target ≥ 4.0/5)
- Classification: TBD (target ≥ 0.90 top-1, 0.96 top-2)
- Latency: TBD (target P50 < 1.5s, P90 < 3s)

## 📝 Reporting
- Export CSV of predictions + metrics
- Add confusion matrix and example wins/failures to the report
- Track versioned prompts/rules for reproducibility

## 🧭 Next Steps
- Expand labeled set, tune prompts/rules, and re-run
- Add per-category metrics to find weak spots
- A/B test IR parameters for better suggestions
