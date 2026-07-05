# Corpus Acceptance Decision

Status: **ACCEPTED — target count met**

The M8 corpus schema, acquisition parser scripts, fixture corpus, corpus-to-ledger loader, and acceptance artifact are implemented. The checked-in public corpus now contains 22 curated public deals:

- 13 EDGAR/REIT-source deals
- 9 CourtListener distressed/foreclosure/enforcement dockets

Acceptance target from `2026-07-04-forecast-lifecycle-m7-m8.md` is 20+ real public deals: 12–15 EDGAR REIT deals and 8–10 CourtListener deals.

Gate artifact: `outputs/results/corpus_acceptance_v1.json`.

Latest gate result:

- `status`: `PASS`
- `deal_count`: 22
- `question_count`: 22
- lifecycle audit: `PASS`
