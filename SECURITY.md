# Security policy

## Supported versions

The latest tagged release receives security fixes. Version 0.1 is an alpha API and may receive breaking changes only through a new minor release.

## Reporting a vulnerability

Use GitHub private vulnerability reporting for sensitive reports. Do not place credentials, proprietary documents, model responses, or exploitable details in a public issue.

The framework does not transmit data unless an explicitly configured target does so. The HTTP adapter sends each question, evidence document, and case metadata to the configured endpoint. Raw target output is excluded from reports unless `--include-raw` is supplied.
