# CODEMAP.md — Universal Codebase Structural Index

> **Read this entire file before doing anything.** It tells you how to build and maintain a structural index of this repository, stored at `.codemap/index.json`. The index is the single source of truth for *what exists in this codebase and how the parts relate*. Code and index must stay in sync — code changes without index updates are bugs.

This protocol is language- and stack-agnostic. It works for full-stack software (frontend + backend + database + infra), research codebases (experiments + datasets + models), monorepos, libraries, and single-language scripts.

## 1. What you must produce

A file at `.codemap/index.json` matching the schema in §3. On every subsequent code change, update both the code and the index in the same turn, following §6 (Maintenance Protocol).

## 2. Top-level structure

```json
{
  "meta":          { ... },
  "layers":        { ... },
  "files":         { ... },
  "symbols":       { ... },
  "relationships": [ ... ],
  "indexes":       { ... }
}
```

## 3. Schema

### 3.1 `meta`
| Field | Type | Notes |
|---|---|---|
| `schema_version` | string | This protocol's version. Current = `"1.0"` |
| `project_name` | string | |
| `project_kind` | enum | `"software"` / `"research"` / `"hybrid"` / `"library"` / `"script"` |
| `last_updated` | ISO-8601 | Update on every change |
| `updated_by` | string | Tool name (`"copilot"`, `"claude"`, `"cursor"`, `"manual"`) |
| `languages` | list[string] | All source languages present |
| `primary_entry_points` | list[symbol_id] | Where execution begins |

### 3.2 `layers`
A dictionary of the logical layers in this project. Use only layers that apply.

Standard layer keys (use these names exactly):
- `frontend` — UI code (React, Vue, Svelte, mobile, etc.)
- `backend` — server-side application code
- `database` — schemas, migrations, queries, stored procedures
- `infra` — IaC, Docker, K8s, CI/CD, deployment configs
- `ml_model` — model architecture, training, inference code
- `data_pipeline` — ETL, data prep, feature engineering
- `research` — experiments, evaluation scripts, paper code
- `shared` — cross-layer libraries and utilities
- `test` — test suites
- `docs` — documentation
- `config` — environment, settings, secrets references

Each layer entry:
```json
"backend": {
  "description": "FastAPI service exposing /api/v1.",
  "root_paths": ["src/api/", "src/services/"],
  "primary_language": "python",
  "frameworks": ["fastapi", "sqlalchemy"]
}
```

### 3.3 `files`
Keyed by relative path (use forward slashes regardless of OS).

| Field | Type | Notes |
|---|---|---|
| `language` | string | |
| `layer` | string | Must match a key in `layers` |
| `purpose` | string | One-line semantic summary |
| `exports` | list[symbol_id] | Symbols defined in this file |
| `imports_internal` | list[file_path] | |
| `imports_external` | list[string] | Package/module names |
| `tags` | list[string] | Free-form labels |
| `loc` | integer | Line count |

### 3.4 `symbols`
Keyed by symbol ID (see §4). **Universal fields required for every symbol:**

| Field | Type | Notes |
|---|---|---|
| `kind` | enum | See §5 |
| `name` | string | |
| `file` | string | Relative path |
| `lines` | `[int, int]` | `[start, end]` |
| `layer` | string | Which layer this belongs to |
| `purpose` | string | One-line semantic description: *what it does, not how* |
| `tags` | list[string] | |
| `semantic_signature` | string | Hyphenated, 3–6 lowercase tokens capturing intent (e.g. `user-authentication-token-validator`). This is the deduplication key. |

**Kind-specific fields** add on top of the universal ones. See §5.

### 3.5 `relationships`
Append-only list of typed edges:

```json
{"from": "<symbol_id>", "to": "<symbol_id>", "type": "<rel_type>"}
```

Allowed `type` values (use exactly these strings):

| Type | Direction | Use for |
|---|---|---|
| `imports` | file → file | Module imports |
| `calls` | function → function | Function/method invocation |
| `inherits` | class → base | OO inheritance |
| `implements` | class → interface | Interface implementation |
| `instantiates` | class → class | `new X()` / `X()` calls |
| `queries` | code → db_table | Read-only DB access |
| `writes_to` | code → db_table | Insert/update/delete |
| `consumes_endpoint` | frontend → api_endpoint | UI calls API |
| `serves_endpoint` | function → api_endpoint | Handler registered for route |
| `triggers` | event → handler | Event/queue/cron |
| `reads_config` | code → config | Reads env var or setting |
| `produces` | pipeline → dataset/artifact | Pipeline output |
| `consumes` | pipeline → dataset/artifact | Pipeline input |
| `tests` | test → target | Test covers symbol |
| `deploys` | infra → service | IaC deploys runtime |
| `depends_on` | infra → infra | Infra resource dependency |

Never store reverse edges — they are derivable. If you need `called_by`, compute it from `relationships`.

### 3.6 `indexes`
Derived lookup structures. Rebuild whenever symbols change.

| Index | Shape |
|---|---|
| `by_layer` | `{layer_name: [symbol_ids]}` |
| `by_tag` | `{tag: [symbol_ids]}` |
| `by_semantic_signature` | `{signature: [symbol_ids]}` |
| `entry_points` | `[symbol_ids]` |
| `cross_layer_paths` | `[{name, path: [symbol_ids]}]` — e.g. `frontend → backend → db` chains |
| `potential_duplicates` | `[{symbols, reason, suggested_action}]` |

## 4. ID convention

Format: `<kind>:<path>:<qualified_name>`

| ID | Example |
|---|---|
| File | `file:src/api/users.ts` |
| Class | `class:src/models/user.py:User` |
| Function | `fn:src/utils/auth.ts:validateToken` |
| Method | `method:src/models/user.py:User.save` |
| UI component | `component:web/src/Login.tsx:LoginForm` |
| API endpoint | `endpoint:src/api/users.py:GET /api/v1/users/:id` |
| DB table | `table:db/schema.sql:users` |
| DB migration | `migration:db/migrations/20260115_add_email.sql:add_email_to_users` |
| Config | `config:.env.example:DATABASE_URL` |
| Infra resource | `infra:terraform/main.tf:aws_rds_instance.primary` |
| Experiment | `experiment:experiments/sweep.py:weat_sweep_v3` |
| Dataset | `dataset:data/loader.py:HindiBiasCorpus` |
| Model | `model:models/registry.py:xlmr_large` |
| Type/interface | `type:src/schemas/user.ts:UserProfile` |
| Test | `test:tests/test_users.py:test_get_user_by_id` |

Rules:
- IDs only change on rename or move. All other fields can change freely.
- For endpoints, the qualified name is `METHOD route`.
- For SQL objects in multi-schema DBs, qualify by schema: `table:db/schema.sql:auth.users`.

## 5. Symbol kinds and their fields

For each kind below, list the extra fields beyond the universal ones in §3.4.

### `class` / `interface` / `struct` / `trait` / `type` / `enum`
- `bases` — list of parent symbol IDs
- `responsibilities` — 3–5 short semantic bullets (used for dedup)
- `methods` — list of method symbol IDs
- `fields` — list of `{name, type, visibility}`

### `function` / `method`
- `signature` — full signature as a string
- `params` — list of `{name, type, doc}`
- `returns` — `{type, doc}`
- `calls` — symbol IDs invoked (also mirrored in `relationships`)
- `side_effects` — list of strings (`"writes file"`, `"http call"`, `"db write"`, `"reads env"`, `"pure"`)
- `async` — boolean
- `visibility` — `"public"` / `"private"` / `"internal"`

### `ui_component`
- `framework` — `"react"` / `"vue"` / `"svelte"` / `"angular"` / `"flutter"` / etc.
- `props` — list of `{name, type, required, doc}`
- `state` — list of state variable names
- `consumes_endpoints` — list of endpoint symbol IDs
- `renders` — list of child component IDs

### `api_endpoint`
- `method` — `"GET"` / `"POST"` / `"PUT"` / `"PATCH"` / `"DELETE"` / `"query"` / `"mutation"` / `"subscription"` (GraphQL) / `"rpc"`
- `route` — path string
- `handler` — function symbol ID
- `request_schema` — type symbol ID
- `response_schema` — type symbol ID
- `auth_required` — boolean
- `queries_tables` — list of DB table symbol IDs
- `rate_limit` — optional string

### `db_table` / `db_view`
- `database` — logical DB name (e.g. `"primary"`, `"analytics"`)
- `columns` — list of `{name, type, nullable, indexed, fk}`
- `primary_key` — list of column names
- `indexes` — list of `{name, columns, unique}`
- `migrations` — list of migration symbol IDs that touched this table
- `referenced_by` — code symbol IDs that read/write this table

### `db_migration`
- `version` — migration ID or timestamp
- `direction` — `"up"` / `"down"` / `"reversible"`
- `affects_tables` — list of table symbol IDs
- `breaking` — boolean

### `config`
- `key` — config key name
- `type` — `"env_var"` / `"yaml_key"` / `"toml_key"` / `"json_key"`
- `default` — default value or `null`
- `required` — boolean
- `consumers` — symbol IDs that read this config
- **Never store the actual secret value — only the key name.**

### `infra_resource`
- `provider` — `"terraform"` / `"pulumi"` / `"cdk"` / `"k8s"` / `"helm"` / `"docker"` / `"github_actions"`
- `resource_type` — provider-specific (e.g. `"aws_rds_instance"`, `"kubernetes_deployment"`)
- `depends_on` — list of other infra symbol IDs

### `experiment` (research)
- `hypothesis` — one-line testable statement
- `script` — entry-point file path
- `dataset_refs` — dataset symbol IDs
- `model_refs` — model symbol IDs
- `metrics` — list of metric names
- `seeds` — list of seeds used
- `hyperparams` — dict
- `status` — `"draft"` / `"running"` / `"complete"` / `"abandoned"`

### `dataset` (research)
- `source` — local path, URL, or Hub ID (e.g. `"hf://Debk/Multi-CrowS-Pairs"`)
- `size` — string (e.g. `"12k examples"`)
- `splits` — list (`"train"`, `"val"`, `"test"`)
- `license` — string

### `model_artifact` (research)
- `architecture` — string
- `params_count` — string (e.g. `"560M"`)
- `checkpoint_path` — string
- `trained_on` — dataset symbol IDs

### `test`
- `test_kind` — `"unit"` / `"integration"` / `"e2e"` / `"property"` / `"benchmark"`
- `covers` — symbol IDs being tested

### Language → kind mapping cheatsheet

| File pattern | Likely kinds | Default layer |
|---|---|---|
| `*.py` | `class`, `function`, `method` | varies — read imports |
| `*.ts` / `*.tsx` / `*.js` / `*.jsx` | `function`, `class`, `type`, `interface`; `ui_component` if returns JSX | frontend or backend |
| `*.vue` / `*.svelte` | `ui_component` | frontend |
| `*.go` | `function`, `method`, `struct`, `interface` | backend |
| `*.rs` | `function`, `struct`, `enum`, `trait`, `impl` | backend / systems |
| `*.java` / `*.kt` | `class`, `interface`, `method`, `enum` | backend |
| `*.cs` | `class`, `interface`, `method`, `enum` | backend |
| `*.rb` | `class`, `module`, `method` | backend |
| `*.php` | `class`, `function`, `method` | backend |
| `*.swift` / `*.kt` (Android) | `class`, `struct`, `function`, `ui_component` | frontend (mobile) |
| `*.sql` | `db_table`, `db_view`, `db_migration`, `db_procedure` | database |
| `*.tf` / `*.tfvars` | `infra_resource` | infra |
| `*.yaml` under `k8s/`, `helm/`, `kustomize/` | `infra_resource` | infra |
| `Dockerfile`, `docker-compose.yml` | `infra_resource` | infra |
| `.github/workflows/*.yml`, `.gitlab-ci.yml` | `infra_resource` | infra |
| `.env*`, `*.config.*`, `settings.py` | `config` | config |
| `*test*.py`, `*.spec.ts`, `*_test.go`, `*Test.java` | `test` | test |

If a single file declares both a route and its handler (FastAPI, Express, Flask, Gin, etc.), create both an `api_endpoint` symbol and a `function` symbol, linked via `serves_endpoint`.

## 6. Bootstrap procedure (first time on a repo)

Run in order. Do not skip phases.

### Phase 1 — Reconnaissance
1. List top-level directories and key files.
2. Detect languages from extensions.
3. Detect layers from directory names (`frontend/`, `client/`, `web/`, `api/`, `backend/`, `server/`, `db/`, `migrations/`, `infra/`, `terraform/`, `k8s/`, `experiments/`, `notebooks/`, `tests/`, `docs/`) and from frameworks (`package.json`, `requirements.txt`, `go.mod`, `Cargo.toml`, `pom.xml`).
4. Write `meta` and `layers` to `.codemap/index.json`. Save.

### Phase 2 — File inventory
Walk every source file. Respect `.gitignore`. Skip generated artifacts (`dist/`, `build/`, `node_modules/`, `__pycache__/`, `target/`, `.next/`, `*.pyc`). For each file, create a `files` entry. If you cannot yet resolve imports, leave those fields empty and fill them in Phase 4. Save.

### Phase 3 — Symbol extraction
Process files in this order:
1. Entry points (`main.*`, `app.*`, `index.*`, `__main__.py`, `cmd/*/main.go`)
2. Files imported by entry points
3. Remaining files breadth-first by import depth
4. Database schemas and migrations
5. Infra files
6. Tests

For each symbol, fill **all** required fields. If a value genuinely cannot be determined, write `"unknown"` — never omit a required field. Save after every ~50 symbols so progress is not lost.

### Phase 4 — Relationships
With all symbols in place, walk the code again and emit edges:
- Function/method invocations → `calls`
- DB queries (raw SQL strings, ORM calls) → `queries` / `writes_to`
- Frontend `fetch` / `axios` / generated-client calls → `consumes_endpoint`
- Route registrations → `serves_endpoint`
- `os.environ`, `process.env`, settings reads → `reads_config`
- Data-pipeline IO → `produces` / `consumes`
- Test targets → `tests`

### Phase 5 — Indexes
Build `by_layer`, `by_tag`, `by_semantic_signature`, `entry_points`, and `cross_layer_paths` (look for chains like `ui_component → consumes_endpoint → api_endpoint → serves_endpoint → fn → queries → db_table`).

### Phase 6 — Duplication scan
For every pair of same-`kind` symbols:
- Exact `semantic_signature` match → flag
- ≥ 60% overlap in `responsibilities` (semantic, not lexical) → flag
- Same `signature` + similar `purpose` → flag
- Same `kind` + same `tags` + heavily overlapping `calls` → flag

Write findings to `indexes.potential_duplicates`. Do not auto-merge. Each entry must include `symbols`, `reason`, and `suggested_action`.

### Phase 7 — Validation
- Every ID referenced in `relationships`, `files.exports`, or `indexes` exists in `symbols` or `files`.
- Every symbol's `file` exists in `files`.
- JSON parses.
- `meta.last_updated` is current.

Report what you built, what you skipped, and what you flagged.

## 7. Maintenance protocol (every code change)

### 7.1 Before editing
1. Read `.codemap/index.json`.
2. Search by `purpose`, `responsibilities`, `tags`, `semantic_signature` for anything related to the task.
3. Check `indexes.potential_duplicates`.
4. Trace `relationships` to identify downstream impact.

### 7.2 Adding a new symbol
1. Search for similar `semantic_signature` or overlapping `responsibilities` first. If a match exists (≥ 60% responsibility overlap), prefer extending the existing symbol. State this to the user before writing code.
2. Write the code.
3. Append a `symbols` entry with all required fields.
4. Update the file's `exports`.
5. Add `relationships` for `calls`, `queries`, `consumes_endpoint`, `reads_config`, etc.
6. Update `indexes.by_layer`, `by_tag`, `by_semantic_signature`.

### 7.3 Deleting or renaming
1. Remove (or rewrite the ID for) the symbol.
2. Update every `relationship` that references the old ID.
3. Remove from every `files[*].exports` and `indexes.*`.
4. Find symbols whose `calls` list contained the deleted ID — update or flag them.

### 7.4 Modifying behavior
- Semantics changed → update `purpose`, `responsibilities`, `semantic_signature`.
- API surface changed → update `signature`, `params`, `returns`.
- New downstream invocations → update `calls` and add `relationships`.
- Now reads new config / queries new table → add corresponding edges.

### 7.5 Cross-layer changes (the common failure mode)
**Changing an API endpoint signature:**
1. Update the `api_endpoint` symbol.
2. Update the handler function's `signature`.
3. Find frontend components whose `consumes_endpoint` includes this endpoint — update them and their props/state if needed.
4. If request/response schemas changed, update the type symbols on both sides.
5. If a generated client exists, update it.

**Changing a DB column:**
1. Update the `db_table` symbol's `columns`.
2. Create a new `db_migration` symbol (and code).
3. Find every symbol with `queries` or `writes_to` edges to this table — check for breakage. Update queries and ORM models.
4. Update API response schemas if exposed.

**Renaming a config key:**
1. Update the `config` symbol.
2. Update every symbol in `consumers` and the corresponding `reads_config` edges.
3. Update `.env.example` and deployment configs.

### 7.6 Always
- Update `meta.last_updated` and `meta.updated_by`.
- Keep JSON valid and `symbols` keys sorted alphabetically.
- If you cannot fill a required field, write `"unknown"` — never omit.
- Produce a diff of `.codemap/index.json` alongside the code diff. An untouched index after a code change is a bug.

## 8. Duplication detection — the philosophical core

Two symbols are duplicates if they would confuse a future engineer. Detect using:

1. **Exact `semantic_signature` match** — strong signal; almost always merge.
2. **Responsibility overlap ≥ 60%** — strong signal; inspect.
3. **Same `signature` + similar `purpose`** — strong signal.
4. **Same `kind` + same `tags` + heavily overlapping `calls`** — likely refactor target.

When constructing `semantic_signature`:
- Hyphenated, lowercase, 3–6 tokens.
- Describes *what*, not *how*: `user-email-validator` not `regex-checker-using-re-module`.
- Use consistent vocabulary across the repo. If `score`, `compute`, and `calculate` all appear, pick one and stick with it. Same for `fetch`/`get`/`load`/`retrieve`.
- Prefer domain terms over generic verbs: `weat-effect-size-calculator` over `score-computer`.

When flagging in `potential_duplicates`, write `suggested_action` as one of:
- `"merge into <id>"`
- `"extract shared logic into <new_id>"`
- `"keep both — distinct contexts"` (with reason)
- `"review"`

## 9. What NOT to do

- Do not modify code without updating the index in the same turn.
- Do not create a new symbol when an existing one has ≥ 60% responsibility overlap. Extend the existing one.
- Do not invent fields outside this schema. If something doesn't fit, propose a schema change to the user.
- Do not index generated artifacts (`dist/`, `build/`, `node_modules/`, `__pycache__/`, compiled binaries, lockfiles).
- Do not index transient state (logs, caches, `.DS_Store`).
- Do not store secret values in `config` entries — only key names.
- Do not store reverse indexes (`called_by`, `referenced_by` at the function level) — they are derivable from `relationships`. The one exception is `db_table.referenced_by`, which is kept for fast impact analysis.
- Do not let `purpose` drift across passes. Once set, change it only when semantics change.
- Do not bloat `tags`. Aim for 1–4 per symbol. Tags should help filter, not catalog.
- Do not write hedging fillers (`"this function maybe does X"`, `"likely handles Y"`). Either determine the value or write `"unknown"`.

## 10. Examples

### 10.1 Backend Python (FastAPI endpoint)
```json
"endpoint:src/api/users.py:GET /api/v1/users/:id": {
  "kind": "api_endpoint",
  "name": "GET /api/v1/users/:id",
  "file": "src/api/users.py",
  "lines": [42, 58],
  "layer": "backend",
  "purpose": "Returns a user profile by ID for authenticated callers.",
  "method": "GET",
  "route": "/api/v1/users/:id",
  "handler": "fn:src/api/users.py:get_user_by_id",
  "request_schema": "type:src/schemas/user.py:UserIdParam",
  "response_schema": "type:src/schemas/user.py:UserProfile",
  "auth_required": true,
  "queries_tables": ["table:db/schema.sql:users"],
  "tags": ["users", "read"],
  "semantic_signature": "fetch-user-profile-by-id"
}
```

### 10.2 Frontend React/TypeScript component
```json
"component:web/src/UserProfile.tsx:UserProfile": {
  "kind": "ui_component",
  "name": "UserProfile",
  "file": "web/src/UserProfile.tsx",
  "lines": [10, 84],
  "layer": "frontend",
  "purpose": "Renders the current user's profile and lets them edit their bio.",
  "framework": "react",
  "props": [
    {"name": "userId", "type": "string", "required": true, "doc": "ID of the user to show"}
  ],
  "state": ["bio", "isEditing"],
  "consumes_endpoints": [
    "endpoint:src/api/users.py:GET /api/v1/users/:id",
    "endpoint:src/api/users.py:PATCH /api/v1/users/:id"
  ],
  "renders": ["component:web/src/Avatar.tsx:Avatar"],
  "tags": ["users", "profile"],
  "semantic_signature": "user-profile-view-and-edit"
}
```

### 10.3 Database table
```json
"table:db/schema.sql:users": {
  "kind": "db_table",
  "name": "users",
  "file": "db/schema.sql",
  "lines": [3, 18],
  "layer": "database",
  "purpose": "Application user accounts.",
  "database": "primary",
  "columns": [
    {"name": "id",    "type": "uuid",         "nullable": false, "indexed": true,  "fk": null},
    {"name": "email", "type": "varchar(255)", "nullable": false, "indexed": true,  "fk": null},
    {"name": "bio",   "type": "text",         "nullable": true,  "indexed": false, "fk": null}
  ],
  "primary_key": ["id"],
  "indexes": [{"name": "idx_users_email", "columns": ["email"], "unique": true}],
  "migrations": ["migration:db/migrations/20260115_create_users.sql:create_users"],
  "referenced_by": ["fn:src/api/users.py:get_user_by_id"],
  "tags": ["core"],
  "semantic_signature": "user-account-table"
}
```

### 10.4 Research experiment (Python)
```json
"experiment:experiments/weat_sweep.py:weat_sweep_v3": {
  "kind": "experiment",
  "name": "weat_sweep_v3",
  "file": "experiments/weat_sweep.py",
  "lines": [1, 220],
  "layer": "research",
  "purpose": "Sweep WEAT effect sizes across 5 multilingual models on the Hindi bias corpus.",
  "hypothesis": "WEAT effect size grows monotonically with model size in Hindi.",
  "script": "experiments/weat_sweep.py",
  "dataset_refs": ["dataset:data/loader.py:HindiBiasCorpus"],
  "model_refs": [
    "model:models/registry.py:mbert",
    "model:models/registry.py:xlmr_base",
    "model:models/registry.py:xlmr_large"
  ],
  "metrics": ["weat_effect_size", "p_value"],
  "seeds": [42, 1337, 2024],
  "hyperparams": {"batch_size": 32, "n_permutations": 10000},
  "status": "running",
  "tags": ["weat", "multilingual", "sweep"],
  "semantic_signature": "multilingual-weat-effect-sweep"
}
```

### 10.5 Infra (Terraform)
```json
"infra:terraform/db.tf:aws_rds_instance.primary": {
  "kind": "infra_resource",
  "name": "aws_rds_instance.primary",
  "file": "terraform/db.tf",
  "lines": [12, 34],
  "layer": "infra",
  "purpose": "Primary Postgres RDS instance backing the API.",
  "provider": "terraform",
  "resource_type": "aws_rds_instance",
  "depends_on": ["infra:terraform/network.tf:aws_vpc.main"],
  "tags": ["database", "production"],
  "semantic_signature": "primary-postgres-rds"
}
```

### 10.6 Go function
```json
"fn:internal/auth/token.go:ValidateToken": {
  "kind": "function",
  "name": "ValidateToken",
  "file": "internal/auth/token.go",
  "lines": [22, 47],
  "layer": "backend",
  "purpose": "Verifies a JWT and returns the claims, or an error if expired or malformed.",
  "signature": "func ValidateToken(token string) (*Claims, error)",
  "params": [{"name": "token", "type": "string", "doc": "raw JWT"}],
  "returns": {"type": "(*Claims, error)", "doc": "parsed claims or error"},
  "calls": ["fn:internal/auth/keys.go:LoadPublicKey"],
  "side_effects": ["reads env"],
  "async": false,
  "visibility": "public",
  "tags": ["auth", "jwt"],
  "semantic_signature": "jwt-token-validator"
}
```

### 10.7 A potential-duplicate flag
```json
{
  "symbols": [
    "fn:src/bias/old_scorer.py:weat_score",
    "fn:src/bias/scorer.py:BiasScorer.compute_weat"
  ],
  "reason": "Both compute WEAT effect size; identical semantic_signature; 80% overlapping call graph.",
  "suggested_action": "merge into BiasScorer.compute_weat, remove src/bias/old_scorer.py"
}
```

## 11. Final instruction

- When asked to **bootstrap**, run Phases 1–7 end to end and produce `.codemap/index.json`.
- When asked to **maintain** (any code change), follow §7 and produce a diff of `.codemap/index.json` alongside the code diff.
- When asked **anything else about this repo**, read `.codemap/index.json` first and use it to ground your answer.

Treat the index as a load-bearing artifact. If you cannot keep it in sync with a change, stop and tell the user before proceeding.
