# Journal Scope Analysis for PDE-Agents Paper

## Paper Profile

| Dimension | Value |
|-----------|-------|
| Core topic | LLM multi-agent systems for scientific computing |
| Application domain | Finite element analysis (heat equation) |
| Novelty type | System design + empirical evaluation |
| Contribution style | Implementation + ablation + V&V + production metrics |
| Word count (target) | ~8,000 words |
| Figures | 4 (architecture, KG schema, convergence plot, ablation bar) |
| Tables | 3 (V&V convergence, ablation results, agent metrics) |
| Code/reproducibility | Full open-source release |

---

## Recommended Journals (ranked)

### 1. ★★★ Engineering with Computers (Springer)
- **ISSN**: 0177-0667 | **Impact Factor**: ~8.7 (2023)
- **Scope**: *"Cutting-edge research in computational science and engineering."*
  Explicitly covers AI/ML methods applied to engineering simulation, mesh generation,
  solver automation. Regular special issues on AI+FEM.
- **Why this paper fits**: The paper is exactly the intersection Springer EwC publishes:
  an AI system (LLM agents) applied to an engineering computation task (FEM), with
  rigorous empirical evaluation. The V&V section satisfies their emphasis on
  correctness verification.
- **Format**: Two-column article, 6000–10,000 words, LaTeX (Springer svjour3 class)
- **Review time**: ~3–4 months
- **Open Access option**: Yes (CC BY, ~€2,790 APC)
- **Submission URL**: springer.com/journal/366
- **Decision**: **Primary target** — best subject-matter fit.

---

### 2. ★★★ Advanced Engineering Informatics (Elsevier)
- **ISSN**: 1474-0346 | **Impact Factor**: ~8.8 (2023)
- **Scope**: *"Intelligent and digital engineering: AI, knowledge management,
  decision support, design automation."* Frequently publishes papers on
  knowledge graphs for engineering, agent-based systems, and LLM applications
  in industrial contexts.
- **Why this paper fits**: The KG-augmented reasoning and multi-agent orchestration
  are core AEI themes. The "knowledge management" angle (Neo4j KG, document
  ingestion pipeline) is a strong fit. The ablation study on KG contribution is
  exactly the kind of empirical AI analysis the journal favours.
- **Format**: Single-column article, up to 12,000 words, LaTeX (Elsevier elsarticle class)
- **Review time**: ~2–3 months
- **Open Access option**: Yes (CC BY, ~$3,500 APC)
- **Submission URL**: elsevier.com/locate/aei
- **Decision**: **Strong alternative** — slightly higher IF, broader scope.

---

### 3. ★★ Computer Methods in Applied Mechanics and Engineering (CMAME, Elsevier)
- **ISSN**: 0045-7825 | **Impact Factor**: ~6.9 (2023)
- **Scope**: *"Computational mechanics, numerical methods, and their applications."*
  Top-tier FEM journal, but the AI/orchestration aspects are secondary.
- **Why this paper fits (partially)**: The V&V study and DOLFINx solver are strong
  CMAME content. However, the LLM/agent focus may be seen as peripheral.
  Best path: strengthen the numerical analysis sections and frame the paper
  primarily as a FEM automation tool.
- **Format**: Double-column, 6000–8000 words, Elsevier elsarticle
- **Review time**: ~4–6 months (longer, prestigious journal)
- **Open Access option**: Yes (~$3,900 APC)
- **Decision**: **Conditional** — submit here only if reframing toward FEM automation.
  Requires toning down the "LLM agent" framing in favour of "automated FEM methodology."

---

### 4. Honorable Mentions

| Journal | IF | Fit | Notes |
|---------|----|-----|-------|
| Scientific Reports (Nature) | ~4.6 | Medium | Broad scope, fast review; lower prestige for computational engineering |
| npj Computational Materials | ~9.7 | Low–Medium | Excellent IF but scope is materials discovery, not simulation automation |
| Journal of Computational Science (Elsevier) | ~3.8 | Medium | Good fit but lower IF; good fallback |
| JAAMAS (Springer) | ~2.9 | Medium | Multi-agent systems focus but misses the FEM contribution |
| Computers & Structures (Elsevier) | ~4.4 | Low | Very traditional FEM, unlikely to value LLM framing |

---

## Conference Options (if journal takes too long)

| Venue | Deadline | Notes |
|-------|----------|-------|
| NeurIPS 2025 Workshop on AI for Science | ~Oct 2025 | Ideal for the ablation finding |
| ECCOMAS Congress 2026 | ~Jan 2026 | FEM community; good visibility |
| SciPy 2026 | ~Mar 2026 | Open-source scientific Python community |
| ICLR 2026 Workshop | ~Jan 2026 | High visibility for AI-science work |

---

## Recommended Strategy

1. **Submit to Engineering with Computers** first (best fit, fast review).
2. While under review, prepare a shorter version (4 pages) for the
   **NeurIPS AI4Science workshop** — especially the KG ablation finding,
   which is a standalone interesting negative result for the ML community.
3. If EwC rejects, revise for **Advanced Engineering Informatics**.

---

## Format Change Required per Journal

| Journal | Class file | Columns | Word limit | Key changes |
|---------|-----------|---------|------------|-------------|
| Eng. with Computers | `svjour3` | 2 | ~8,000 | Springer template, Author info format |
| Adv. Eng. Informatics | `elsarticle` | 1 | ~10,000 | Elsevier template, Highlights box (5 bullets) |
| CMAME | `elsarticle` | 2 | ~7,000 | Stronger numerical focus, reduce LLM intro |
