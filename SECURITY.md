# Security and account information

Do not place broker passwords, account identifiers, API keys, database passwords,
terminal data folders or model artifacts in this repository or an issue.
Use local environment variables or untracked configuration files where a tool
supports them. Keep `.env` files untracked.

If you discover exposed credentials, notify the repository owner privately
through GitHub rather than posting the value in a public issue. Rotate the
credential at its provider; deleting a file does not revoke a leaked secret.

The publication tool checks its allowlist for credential patterns and rejects
unexpected file types. This is a publishing aid, not a full security assessment.

Native EAs and legacy integration scripts can interact with a trading terminal.
Offline CI does not validate live-order safety. Review symbol specifications,
volume limits, account-wide exposure, timezone resets and account rules before
any separate demo deployment.
