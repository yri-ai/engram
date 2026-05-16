# Branch Promotion and Deployment Flow

This repository now supports a staged promotion flow:

1. `main`/`master` → `staging`
2. `staging` → `production`

## What was added

- `.github/workflows/deploy.yml`
  - Deploys on push to `staging` and `production`
  - Supports manual deploy via `workflow_dispatch`
  - Runs verification tests before deploy
  - Triggers environment-specific deployment webhooks

- `.github/workflows/promote.yml`
  - Manual branch promotion workflow
  - Supported pairs:
    - `main` → `staging`
    - `master` → `staging`
    - `staging` → `production`

- `.github/workflows/ci.yml`
  - CI now runs for `main`/`master`/`staging`/`production`

## Required GitHub setup

### 1) Create branches

- `staging`
- `production`

If you use `main`, keep `main` as your dev integration branch.
If you still use `master`, that is also supported.

### 2) Create environments

In **Settings → Environments**, create:

- `staging`
- `production`

For `production`, enable required reviewers/approval gates.

### 3) Add repository/environment secrets

Required by `deploy.yml`:

- `STAGING_DEPLOY_WEBHOOK_URL`
- `PRODUCTION_DEPLOY_WEBHOOK_URL`

These should be deploy endpoints for your platform (Cloud Run trigger, Render deploy hook, etc.).

## Day-to-day usage

### Promote main/master to staging

Option A (manual git merge):

```bash
git checkout staging
git merge main   # or master
git push origin staging
```

Option B (GitHub Action):

- Run workflow: **Promote Branch**
- `source`: `main` (or `master`)
- `target`: `staging`

This triggers:
- CI on `staging`
- Deploy workflow to `staging`

### Promote staging to production

Option A:

```bash
git checkout production
git merge staging
git push origin production
```

Option B:

- Run workflow: **Promote Branch**
- `source`: `staging`
- `target`: `production`

This triggers:
- CI on `production`
- Deploy workflow to `production`

## Recommended protection rules

For `staging` and `production`:

- Require pull request reviews
- Require status checks to pass (`CI`)
- Restrict direct pushes (especially `production`)
- Require linear history (optional)
