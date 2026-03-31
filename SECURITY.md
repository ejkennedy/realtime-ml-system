# Security Policy

## Supported Use

This repository is a public portfolio project intended for learning, demo, and
reference purposes. It is not packaged or supported as a production-ready
security product.

## Reporting A Vulnerability

Please do not open a public GitHub issue for a suspected security problem.

Instead, report it privately with:

- a short description of the issue
- affected area or file path
- reproduction steps
- potential impact

If you do not have a private reporting path configured on GitHub yet, add one
before public launch. Until then, avoid publishing sensitive proof-of-concept
details in the issue tracker.

## Scope Notes

Particular areas worth handling carefully:

- any embedded credentials or example secrets
- MLflow / Redis / MinIO local defaults
- Docker Compose network exposure
- synthetic-data generation paths that could be mistaken for real financial data

## Response Expectations

Because this is a portfolio project, response times are best-effort rather than
SLA-backed.
