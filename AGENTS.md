# Agent instructions

Do not commit implementation plans or temporary working documents, such as
scratch notes, progress logs, review notes, or drafts.

Keep plans in `.agent/plans/` and other working documents in `.agent/work/`.
Both directories are ignored by Git; do not force-add their contents.

Commit lasting project documentation when it is part of the requested change.
Review staged files before committing to catch accidental working documents.

Comments and docstrings describe what the code does now: no authorship notes,
change history, or references to earlier versions, reviews or experiments.

Keep README.md a concise project introduction with documentation links, never
a changelog, work log or design document.
