# Versioning

Interactive ML Labs currently uses two related but separate version signals.

## Product Release Tags

Git tags describe student-facing product releases of the whole repository.

The next planned product tag is:

```text
v0.0.9a
```

This tag means "student-facing alpha of the guided learning platform." It covers
the unified app, guided paths, docs, issue templates, and repository release
readiness as one product milestone.

## Python Package Versions

The workspace packages currently use:

```toml
version = "0.1.0"
```

This is intentionally left as the technical package version for now. The packages
are not yet published independently, and the student alpha is not distributed as
individual Python package releases.

## Current Decision

For `v0.0.9a`, keep package versions at `0.1.0` and use the Git tag/release as
the source of truth for the student-facing product milestone.

This avoids unnecessary package churn before the project has a real packaging
and distribution story. If later releases publish installable artifacts or Python
packages, the project should revisit whether package versions should track
product release tags more closely.
