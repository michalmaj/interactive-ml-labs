# Contributing

This project follows a lightweight professional workflow.

## Branch naming

Use short, descriptive branch names:

```text
chore/init-repository
chore/configure-uv-workspace
chore/configure-quality-tools
ci/add-github-actions
docs/add-architecture
feat/core-interfaces
feat/gradient-descent-dataset
feat/gradient-descent-algorithm
test/gradient-descent-algorithm
```

## Commit messages

Use conventional-style commit messages:

```text
chore: initialize repository structure
chore: configure uv workspace
ci: add github actions workflow
docs: add project architecture
feat(core): add base algorithm interfaces
test(gradient-descent): add algorithm tests
```

## Pull request structure

Each pull request should include:

```markdown
## Summary

Short description of the change.

## Changes

- Change 1
- Change 2
- Change 3

## How to test

Commands or steps used to verify the change.

## Notes for students

What this pull request demonstrates from a software engineering perspective.
```

## Development rules
- Do not commit directly to `main`.
- Keep pull requests small and focused.
- Prefer readable code over clever code.
- Add tests for algorithmic logic.
- Keep rendering separate from machine learning logic.
- Every demo must have its own documentation.

