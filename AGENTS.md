# Agent instructions

- Follow [DESIGN.md](DESIGN.md) for architecture and contracts,
  [CONTRIBUTING.md](CONTRIBUTING.md) for development and verification, and
  [demo guidelines](docs/demo-guidelines.md) when adding or revising demos.
- Use the configured human Git identity. Before committing, check
  `git var GIT_AUTHOR_IDENT` and `git var GIT_COMMITTER_IDENT` against
  `git config user.name` and `git config user.email`. If missing or ambiguous,
  ask. Do not override identity, use an agent author, or add agent attribution
  trailers. Preserve other contributors' attribution; rewrite it only when asked.
- Keep plans in `.agent/plans/` and working notes in `.agent/work/`. Both are
  ignored; never force-add or commit them. Commit lasting project documentation
  and review staged files before committing.
- Comments and docstrings describe current behavior, without authorship notes,
  change history or references to earlier reviews and experiments.
- Keep `README.md` a short introduction with links. Put design in `DESIGN.md`,
  contributor instructions in `CONTRIBUTING.md`, demo conventions in
  `docs/demo-guidelines.md`, and the runnable demo catalog in `demos/README.md`.
