# Public Forecast Corpus Schema

The public corpus is a licensed-clean set of deal timelines used to exercise the forecast lifecycle loop without confidential data.

## Default taxonomy

`public_real_estate_milestone_v1` is a closed three-branch taxonomy:

| Branch ID | Meaning |
|---|---|
| `advance_or_close` | Deal advances materially or closes. |
| `reprice_or_restructure` | Deal reprices, restructures, or terms materially change. |
| `terminated_or_failed` | Deal terminates, fails, or is abandoned. |

## `PublicDeal`

Defined in `src/engram/models/corpus.py`:

- `deal_id`
- `source_kind`: `edgar_reit`, `courtlistener`, `recorder`, or `other`
- `evidence_docs[]`: `{doc_id, url, published_at, retrieved_at, text_ref, summary, role}` where `role` is `forecast_evidence` or `resolution_evidence`. The lifecycle loader excludes `resolution_evidence` from forecast dossiers so closing/outcome sources do not leak the answer into the forecast input.
- `milestones[]`: `{at, kind, description}`
- `resolved_branch`: must be a member of the taxonomy
- `resolved_at`: must be on or after every evidence document and milestone
- `branch_taxonomy_id`: defaults to `public_real_estate_milestone_v1`

Tests assert schema validation, branch validation, and chronology validation.
