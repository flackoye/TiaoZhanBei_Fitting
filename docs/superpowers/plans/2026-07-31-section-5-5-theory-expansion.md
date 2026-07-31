# 5.5 Theory Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand the current 5.5 DOCX with deeper theory and analysis while preserving the first three user-adjusted tables and converting the 5.5.6 fitting table into prose.

**Architecture:** Load the user-adjusted DOCX as the editing base, hash the first three table XML elements, insert new styled paragraphs at stable text anchors, remove only the table located inside 5.5.6, and save to a new output file. Verify section structure, formulas, numerical results, table hashes, source preservation, and DOCX package integrity.

**Tech Stack:** Bundled Python, python-docx, OOXML, structural audit.

## Global Constraints

- Preserve the current DOCX and original 5.5 DOCX unchanged.
- Preserve the first three tables byte-for-byte at the XML element level.
- Remove the 5.5.6 fitting table and keep all of its values in prose.
- Do not add a table in 5.5.7.
- Add approximately 20 to 25 theory and analysis paragraphs plus TV and roughness formulas.
- Output `outputs/documents/5.5_反演结果不确定性与可信度评价_理论扩充版.docx`.

---

### Task 1: Surgical DOCX expansion

**Files:**
- Create: `.superpowers/docx_5_5/expand_theory.py`
- Read: `outputs/documents/5.5_反演结果不确定性与可信度评价_修订版.docx`
- Create: `outputs/documents/5.5_反演结果不确定性与可信度评价_理论扩充版.docx`

**Interfaces:**
- Consumes: current paragraph anchors and first-three-table XML hashes.
- Produces: expanded DOCX with exactly three preserved tables.

- [ ] Insert formal theory and analysis paragraphs after approved anchors in the introduction and sections 5.5.1 through 5.5.7.
- [ ] Insert total-variation and roughness equations into 5.5.6.
- [ ] Remove only the table following the 5.5.6 PCHIP paragraph.
- [ ] Save to the new output path.

### Task 2: Structural verification

**Files:**
- Verify: `outputs/documents/5.5_反演结果不确定性与可信度评价_理论扩充版.docx`

**Interfaces:**
- Consumes: expanded DOCX.
- Produces: pass/fail evidence for delivery.

- [ ] Confirm seven numbered sections, exactly three tables, unchanged hashes for all three preserved tables, required formulas and fitting values, increased body-paragraph count, source checksums unchanged, and a valid DOCX ZIP package.
- [ ] Attempt packaged rendering; if `soffice` remains unavailable, disclose structural-only QA in the final response.
