# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Math in Markdown must be GitHub-friendly

README.md (and any Markdown here) is rendered by GitHub's math support, which
is a restricted MathJax subset with Markdown-interference quirks. Rules learned
from actual rendering failures in this repo:

- **Inline math: always use the dollar-backtick form `` $`...`$ ``, never bare
  `$...$`.** GitHub processes Markdown emphasis *before* math, so underscores
  and asterisks inside bare `$...$` spans pair up as italics across spans and
  the math silently fails to render. The `` $`...`$ `` form is a code span
  first, so it is immune. Keep each inline span on a single line.
- **Display math: `$$...$$`.** Multi-line content is fine (`array`, `pmatrix`
  environments render correctly), but leave a blank line before and after the
  block.
- **Only GitHub-allowed macros.** `\operatorname` is rejected with the error
  "The following macros are not allowed" — use `\mathrm` instead. Known-good
  macros already used in this repo: `\det`, `\mathrm`, `\tilde`, `\widetilde`,
  `\frac`, `\prod`, `\sum`, `\big`/`\Big`, `\left`/`\right`, `\blacksquare`,
  `\mapsto`, `\Theta`, `\text`, `\qquad`, and the `array`/`pmatrix`
  environments. When introducing a new macro, prefer plain LaTeX core; if in
  doubt, spell it with `\mathrm{...}`.

## Mathematical claims in the README

Verify identities before adding them: numerically against the actual functions
in `determinant/determinant.py` (e.g. `DPmatrix`), and symbolically with sympy
for small `n` when feasible. Calibrate novelty claims against the papers in
`references/` (untracked, local only) before presenting a result as new.
