# Paper Revision TODO

---

## 1. Restructure Novel Material Experiment into Section 6.1

**Current state**: Section 6.4 (Novel Material Experiment) is a standalone subsection that comes late in the paper after the ablation analysis and discussion.

**Goal**: Merge the novel-material content back into Section 6.1 (Experimental Design) so the reader encounters it as part of the main experiment, not an afterthought.

- Move the three fictional materials (Novidium, Cryonite, Pyrathane) description into the task design portion of 6.1.
- Move MPF and Physics Score definitions into a dedicated **"Evaluation Metrics"** subsection within 6.1, giving them proper focus as a key contribution — not buried inside the novel-material narrative.
- Structure should flow: task design (including novel tasks) -> evaluation metrics (MPF, Physics Score, sensitivity-weighted MPF) -> methodology (frozen KG, 50 tasks, Wilson CIs).

**Critical notes**:
- This restructuring is the right call. Currently the reader hits 2+ pages of ablation results before learning what MPF and Physics Score even mean. Defining metrics before presenting results is standard practice and makes the paper easier to follow.
- However, the sensitivity-weighted MPF (`MPF_w`) may not deserve to live in the main metrics section. It was computed on the old 7-task experiment and isn't used in the 50-task ablation. Two options: (a) recompute `MPF_w` for the 50-task data and include it, or (b) keep it as a brief remark in the error propagation paragraph only. Option (b) is simpler and avoids introducing a metric that doesn't appear in the main results table.
- The error propagation analysis (Table 7, Figure 8) currently lives inside Section 6.4. After restructuring, this should become a separate subsection (e.g., "6.X Error Propagation") that follows naturally from the novel-material results, not be buried inside the experiment design.

---

## 2. After Section 6.3: Narrow Focus to KG On vs KG Smart

**Current state**: Section 6.3 (Analysis and Discussion) discusses all three modes (On/Off/Smart) equally.

**Goal**: At the end of 6.3, after presenting Table 4 results, explicitly pivot the narrative:

- Acknowledge that KG Off wins on success rate and wall time for standard tasks.
- But establish — citing Table 4 novel-task rows — that KG Off fabricates properties and produces physically incorrect results (MPF=0.34, physics=0.59).
- Conclude: "For any deployment where output correctness matters (not just completion), KG-enabled modes are necessary. We therefore focus the remaining analysis on KG On vs KG Smart."
- All subsequent sections (failure analysis, growth experiment, conclusion) should compare only KG On and KG Smart.

**Critical notes**:
- This pivot is the paper's strongest rhetorical move. Right now the narrative is confused — KG Off "wins" on the headline metric (100% SR), which makes the KG look useless to a casual reader. By reframing the question from "does it run?" to "does it run correctly?", the paper's argument becomes much sharper.
- Be careful not to dismiss KG Off too aggressively. A fair treatment: KG Off is the right choice when parameters are explicit and the material is well-known. KG-enabled modes are necessary when the domain includes proprietary/novel materials or when output fidelity matters for downstream engineering decisions. This nuance is what Algorithm 1 (Adaptive KG Mode Selection) already captures — reference it here.
- The pivot also sets up a cleaner narrative for the growth experiment (Section 7.1): since we're now comparing KG On vs KG Smart only, the paired bar chart makes more sense — it shows where *accumulated experience* (Smart's advantage over On) matters.

---

## 3. Fix Table 5 (Novel Material Properties) — Outdated Data

**Current state**: Table 5 (`tab:novel-props`) shows agent-chosen properties from an older 7-task experiment. The data may not match the current 50-task ablation (Table 4).

**Action**:
- Cross-check Table 5 property values against the actual configs from the latest ablation run (`evaluation/results/ablation_v2_results.json`).
- Extract the novel-task configs for KG Off and KG Smart from the v2 ablation data.
- Update Table 5 with the correct property values from the 50-task run.
- Ensure the caption and body text reference counts are consistent (10 novel tasks in the ablation, 7 in the standalone experiment — decide which to use).

**Critical notes**:
- The v2 ablation results JSON has `config_produced` fields but no `tool_calls_log` or `raw_response`. We need to check if `config_produced` contains the agent's chosen k/rho/cp values. If not, we may need to look up the configs from the result directories on disk (inside the Docker container) using the `run_id` field.
- There's a count mismatch we must resolve: the ablation has 10 novel tasks, the standalone experiment had 7. Table 5 should use the 10-task ablation data since that's what Table 4 references. The 7-task experiment should either be dropped or clearly labelled as a preliminary study.
- KG Off produces *different* fabricated values each run (the LLM hallucinates differently). Table 5 currently shows ranges ("10-45" for Novidium k). With the 10-task data we'll have more data points — showing ranges is honest, or we could show the median fabricated value.

---

## 4. Deep Failure Analysis: KG On vs KG Smart

**Current state**: Section 6.3 mentions iteration-budget exhaustion and warning-induced conservatism as reasons KG On underperforms, but this is hand-wavy — no concrete evidence from logs.

**Goal**: Trace real failures from the ablation logs to build a rigorous evidence base.

### 4a. Extract failure data
- From `evaluation/results/ablation_v2_results.json`, identify all KG On failures (12/50) and KG Smart failures (3/50).
- For each failure: task ID, difficulty, number of iterations used, last tool called, whether `run_simulation` was ever called.

### 4b. Classify failure modes
Build a taxonomy:
- **Budget exhaustion**: agent hit max_iterations without calling `run_simulation`
- **Warning abort**: agent stopped after `check_config_warnings` returned a warning
- **Tool misuse**: agent called non-existent tools or malformed JSON
- **Simulation error**: `run_simulation` was called but failed (solver divergence, mesh error, etc.)

### 4c. Trace KG Smart's advantage
For the tasks where KG On fails but KG Smart succeeds:
- Did KG Smart's warm-start injection provide the right config context upfront?
- Did KG Smart avoid the mandatory KG query overhead that exhausted KG On's budget?
- Quantify: how many of KG On's failures are attributable to (a) warm-start bypass vs (b) lazy retrieval?
- This answers the question: "Is the value in the warm-start, the lazy retrieval, or both?"

### 4d. Present as a table or figure
- Table: failure mode classification for KG On (12 failures) with counts
- Comparison: for each KG On failure, show what KG Smart did differently on the same task
- Quote 1-2 concrete examples in the text (e.g., "Task M07: KG On spent 8 iterations on KG queries and hit the budget at iteration 15 without calling run_simulation. KG Smart received a warm-start config from a similar copper simulation and called run_simulation at iteration 3.")

**Critical notes**:

### Data gap — this is the biggest risk item
The current ablation results JSON does **not** contain `tool_calls_log`, `raw_response`, or `answer` fields. The `agent_iterations` field exists but the detailed tool-call trace (which tool was called at each step) was not captured. This means we cannot currently classify failure modes from the saved data alone.

**Options to fix**:
1. **Re-run the ablation with enhanced logging** — modify `run_ablation_v2.py` to capture the full tool-call history per task. This is the cleanest approach but costs ~1 hour of compute. Since we're only re-running KG On (12 failures to investigate) and KG Smart (3 failures), we could run just those ~15 tasks with verbose logging rather than all 150.
2. **Check PostgreSQL agent logs** — the database stores agent reasoning steps (`agent_logs` table). Query for the specific run_ids from the ablation. However, KG On failures often have `run_id=None` (10 of 12 failures), meaning no simulation was ever launched, so there may be no database entry to query. The agent's LangGraph state is not persisted to DB.
3. **Reconstruct from the `iterations` field** — we know KG On failures averaged ~8 iterations with no `run_id`. Since the max budget was 25 iterations (we raised it from 15), 8 iterations suggests the agent stopped early — likely producing a text response that the router treated as "finish" rather than making a tool call. This points to budget exhaustion as the primary cause, which aligns with our prompt-tightening fix.

**Recommendation**: Option 1 is best. Re-run the 12 KG On failure tasks + 3 KG Smart failure tasks with enhanced logging that captures the full tool-call sequence. This gives us concrete evidence for the paper.

### Decomposing warm-start vs lazy retrieval
This is the most interesting analytical question. To properly answer "which component helps more?", we'd ideally need a 4th mode: **KG Lazy-only** (lazy retrieval without warm-start). Comparing:
- KG On (mandatory queries) vs KG Smart (warm-start + lazy) shows the combined benefit
- KG On vs KG Lazy-only would isolate the lazy retrieval benefit
- KG Lazy-only vs KG Smart would isolate the warm-start benefit

Without running a 4th mode, we can still make reasonable inferences:
- If KG On failures are primarily budget-exhaustion (agent never reaches `run_simulation`), then the warm-start is the key factor — it front-loads context so the agent doesn't waste iterations on KG queries.
- If KG On failures are primarily quality failures (simulation runs but produces bad results), then lazy retrieval is the key factor — it provides corrective feedback on failure.
- From the data: 10/12 KG On failures have `run_id=None` (no simulation ever ran), strongly suggesting budget exhaustion. This points to **warm-start as the dominant factor**.

### KG Smart's 3 failures deserve scrutiny too
- E07 (easy), H04 (hard), N08 (novel) — these fail in KG Smart.
- E07 is easy, so this is unexpected. Could be a flaky LLM response or a genuinely tricky task despite its "easy" label.
- N08 (novel) failing in KG Smart is concerning since KG Smart should handle novel materials well. Worth investigating whether the warm-start retrieved the right material or missed it.
- These same tasks should be checked against KG On and KG Off results: if they fail across all modes, they may be inherently problematic tasks (ambiguous description, solver limitation, etc.).

---

## 5. Strengthen the Conclusion

**Current state**: The conclusion summarises findings but doesn't end on a forward-looking high note about the agentic framework's potential.

**Goal**: End with a compelling argument for KG Smart + agentic workflows.

- Restate KG Smart's key achievement: highest quality (physics 0.918, MPF 0.898) with near-KG-Off reliability (94%) — the best of both worlds.
- Emphasise the architectural insight: the value is in the *integration pattern* (warm-start + lazy retrieval), not just having a knowledge graph.
- Broaden to agentic frameworks: argue that the warm-start + lazy-retrieval pattern generalises beyond FEM — any domain with a growing knowledge base (materials science, drug discovery, circuit design) could benefit from this pattern.
- Final sentence should be aspirational: position PDE-Agents as a template for how autonomous engineering agents should interact with structured knowledge, and note that as KGs grow and LLMs improve, the quality gap between KG-enabled and KG-free agents will only widen for novel/proprietary domains.

**Critical notes**:
- The conclusion currently doesn't mention the KG growth experiment finding at all (we just added it in the previous revision). Make sure the difficulty-dependent growth result is woven in.
- Avoid overclaiming. The 6% success rate gap (94% vs 100%) is real and the paper should acknowledge it honestly: "KG Smart is not yet as reliable as the KG-free baseline on well-characterised tasks, but it is the only mode that delivers both high reliability and high fidelity."
- The "generalises beyond FEM" argument is compelling but needs to be careful. Our evidence is from one domain (heat transfer). State it as a hypothesis supported by architectural reasoning, not as a proven claim. E.g., "We hypothesise that the warm-start + lazy-retrieval pattern generalises to other tool-using agent domains where a growing knowledge base provides task-relevant context."
- Consider adding a brief "lessons learned" paragraph before the final aspirational statement. Something like: "Three design principles emerged: (1) never make KG access mandatory in the agent's critical path, (2) front-load context via embedding similarity rather than requiring the agent to formulate queries, (3) let the agent decide when it needs more knowledge rather than forcing it." These are concrete, actionable takeaways that reviewers will appreciate.

---

## Suggested Execution Order

1. **Item 4 first** (failure analysis) — this requires running experiments and produces data that feeds into items 2 and 5.
2. **Item 3** (fix Table 5) — data extraction, straightforward.
3. **Item 1** (restructure) — large text reorganisation, but no new data needed.
4. **Item 2** (narrow focus) — depends on item 4's findings for the pivot argument.
5. **Item 5** (conclusion) — write last, since it synthesises everything above.

Items 1 and 3 can be done in parallel. Item 4 is on the critical path.
