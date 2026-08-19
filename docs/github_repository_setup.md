# GitHub Repository Setup

This document captures the repository settings recommended for the `v0.0.9a`
student-facing alpha.

Current status: description, topics, Issues, and Wiki were applied on GitHub for
the alpha release preparation. The website field is intentionally empty until a
public project site exists.

## Description

Use this short repository description:

```text
Interactive visual machine learning labs with guided lessons, tasks, progress, and bilingual course flow.
```

## Website

Leave the website field empty for `v0.0.9a` unless a public GitHub Pages site or
project website is created before the release.

The repository README and student alpha notes are the current source of truth:

- [README.md](../README.md),
- [student_alpha_v0_0_9a.md](student_alpha_v0_0_9a.md),
- [student_alpha_v0_0_9a.pl.md](student_alpha_v0_0_9a.pl.md).

## Topics

Recommended topics:

```text
machine-learning
education
pygame
interactive-learning
python
teaching
visualization
ml-education
```

## Home Page Sections

Recommended settings for the GitHub repository home page:

- Releases: enabled and useful for `v0.0.9a`,
- Deployments: leave disabled/unused until there is a deployed website,
- Packages: leave disabled/unused until packaged student builds exist.

## Repository Features

Recommended features for the alpha:

- Issues: enabled, with issue forms for student and classroom feedback,
- Wiki: enabled or prepared for instructor/student notes,
- Discussions: optional later, after the first classroom feedback round.

## CLI Setup

The description, topics, issues, and wiki can be set with GitHub CLI:

```bash
gh repo edit michalmaj/interactive-ml-labs \
  --description "Interactive visual machine learning labs with guided lessons, tasks, progress, and bilingual course flow." \
  --enable-issues \
  --enable-wiki \
  --add-topic machine-learning \
  --add-topic education \
  --add-topic pygame \
  --add-topic interactive-learning \
  --add-topic python \
  --add-topic teaching \
  --add-topic visualization \
  --add-topic ml-education
```

The home-page section visibility may still need a quick manual check in the
GitHub repository settings UI.
