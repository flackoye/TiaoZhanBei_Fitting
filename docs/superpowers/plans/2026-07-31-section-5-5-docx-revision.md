# 5.5 Experiment-Integrated DOCX Revision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a clean revised 5.5 DOCX that preserves the source document's project-book structure while replacing unsupported theory with the verified three-method experiment, selected method, confidence weights, fitting interface, and limitations.

**Architecture:** Use the original `5.5.docx` as the formatting and section-geometry authority. Build a new document from that package after clearing body content, then add seven numbered sections, core equations as centered mathematical text, and four compact evidence tables. Validate content programmatically against the approved numeric constants, inspect OOXML structure, and attempt DOCX-to-PNG rendering with the bundled document renderer.

**Tech Stack:** Bundled Python 3, `python-docx`, OOXML helpers, packaged `render_docx.py`, PowerShell verification.

## Global Constraints

- Do not overwrite `E:\DeskTop\preparation\Chen_Wei\TiaoZhanBei\5.5.docx`.
- Output to `E:\DeskTop\preparation\Chen_Wei\TiaoZhanBei\Fitting\outputs\documents\5.5_反演结果不确定性与可信度评价_修订版.docx`.
- Preserve the source A4 page geometry, margins, Chinese body style, and seven-section hierarchy.
- Report fusion performance using the best fixed coefficient combination over 13 folds, not the average of all 273 candidate records.
- Do not interpret the stress maximum weight of 0.567 as poor model agreement.
- Do not claim that one-group weighted fitting proves a full-dataset significant improvement.
- Keep damage and stress uncertainty and weights separate for the 5.6 interface.

---

### Task 1: Build the revised DOCX from the source document

**Files:**
- Create: `.superpowers/docx_5_5/build_revision.py`
- Read: `E:\DeskTop\preparation\Chen_Wei\TiaoZhanBei\5.5.docx`
- Read: `docs/5.5_uncertainty_experiment_results.md`
- Create: `outputs/documents/5.5_反演结果不确定性与可信度评价_修订版.docx`

**Interfaces:**
- Consumes: the source DOCX styles and page geometry plus the approved chapter design in `docs/superpowers/specs/2026-07-31-section-5-5-experiment-integrated-revision-design.md`.
- Produces: a clean DOCX containing sections 5.5.1 through 5.5.7, four evidence tables, and equations needed to explain aggregation, calibration, evaluation, and fitting weights.

- [ ] **Step 1: Inspect source style names and document body blocks**

Run the bundled Python executable to list paragraph styles, table styles, page geometry, headers, footers, and existing body block counts. Expected: one A4 section, 99 paragraphs, no tables, and reusable `Normal`, `Heading 2`, and `Heading 4` styles.

- [ ] **Step 2: Create the deterministic builder**

Implement `build_revision.py` with these exact responsibilities:

1. Load the source DOCX and preserve section properties, headers, footers, theme, and styles.
2. Remove body paragraphs and tables while retaining `w:sectPr`.
3. Configure Chinese fonts explicitly for body and heading runs.
4. Add the chapter introduction and sections 5.5.1 to 5.5.7 in the approved order.
5. Add equations as centered Cambria Math paragraphs with accompanying variable explanations.
6. Add tables for data profile, three-method CV comparison, selected weight distribution, and one-group fitting comparison.
7. Set explicit table widths, repeat header rows, cell margins, header shading, vertical centering, and deliberate numeric alignment.
8. Save only to the specified output path.

- [ ] **Step 3: Execute the builder**

Run:

```powershell
& 'C:\Users\20900\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' '.superpowers\docx_5_5\build_revision.py'
```

Expected: the revised DOCX exists, is non-empty, and the source DOCX timestamp and checksum remain unchanged.

### Task 2: Validate document content and structure

**Files:**
- Read: `outputs/documents/5.5_反演结果不确定性与可信度评价_修订版.docx`
- Create: `.superpowers/docx_5_5/content_audit.txt`

**Interfaces:**
- Consumes: revised DOCX from Task 1.
- Produces: an audit confirming headings, required data, tables, formulas, and forbidden claims.

- [ ] **Step 1: Extract paragraphs and tables**

Use `python-docx` to write paragraph text, style names, and all table cells to `.superpowers/docx_5_5/content_audit.txt`.

- [ ] **Step 2: Verify mandatory content**

Confirm the document contains:

```text
286,243
96,281
13 折
7.39
0.382
1.76
0.378
(0.8, 0.2, 0.0)
(0.0, 1.0, 0.0)
0.409
0.965
0.379
0.567
damage_weight
stress_weight
Hampel
Savitzky-Golay
PCHIP
```

Expected: every required string is present.

- [ ] **Step 3: Verify forbidden or misleading content is absent**

Confirm that the final chapter does not claim MC Dropout, OOD, conformal prediction, full-dataset fitting improvement, or poor stress model agreement as completed results. Expected: no unsupported implementation claim and no “全量显著提升” statement.

- [ ] **Step 4: Verify structure**

Confirm exactly one chapter heading, seven numbered subsection headings, four tables, no empty accidental paragraphs longer than layout spacing requires, and no source-file overwrite.

### Task 3: Render and visually inspect the deliverable

**Files:**
- Read: `outputs/documents/5.5_反演结果不确定性与可信度评价_修订版.docx`
- Create internally: `.superpowers/docx_5_5/render/page-<N>.png`

**Interfaces:**
- Consumes: structurally validated DOCX.
- Produces: final visual QA result or an explicit LibreOffice-unavailable fallback record.

- [ ] **Step 1: Attempt packaged rendering**

Run:

```powershell
& 'C:\Users\20900\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' 'C:\Users\20900\.codex\plugins\cache\openai-primary-runtime\documents\26.727.11326\skills\documents\render_docx.py' 'outputs\documents\5.5_反演结果不确定性与可信度评价_修订版.docx' --output_dir '.superpowers\docx_5_5\render'
```

Expected: PNGs for every page. If `soffice` is unavailable, record that exact limitation and continue with the structural audit permitted by the document skill.

- [ ] **Step 2: Inspect every rendered page when available**

Check all pages at full resolution for clipped Chinese text, equation wrapping, table overflow, broken page breaks, excessive blank space, font substitutions, and header/footer displacement.

- [ ] **Step 3: Iterate if necessary**

If a defect is visible, update only `.superpowers/docx_5_5/build_revision.py`, rebuild, rerun the content audit, and rerender until clean.

### Task 4: Final verification and handoff

**Files:**
- Verify: `outputs/documents/5.5_反演结果不确定性与可信度评价_修订版.docx`

**Interfaces:**
- Consumes: final audited DOCX.
- Produces: one user-facing DOCX deliverable.

- [ ] **Step 1: Verify source preservation and output metadata**

Confirm the source checksum is unchanged, output opens through `python-docx`, has one section, seven numbered subsections, four tables, and nonzero size.

- [ ] **Step 2: Report only the final DOCX**

Return the final DOCX with one output citation. Mention representative revisions and disclose only if LibreOffice rendering was unavailable; do not expose internal scripts, audit files, PNGs, or PDFs unless requested.
