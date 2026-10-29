# Acceptance runner checks

Seven unit tests passed. They check rejection of missing/stale/failed gates, exact source membership and both required sandbox probes. The subprocess termination test uses an explicitly synthetic compiler process, not a proof result.

The seven official-comparator integration tests did **not** pass. Their real commands are blocked before replay by the host systemd/Landlock requirements; diagnostics are preserved in comparator-integration.log. The missing-comparator rejection case passes. The fixture now copies the port-layout helper required by the actual gate scripts. No mock replaces real replay.

Sources, commands, exit statuses, logs and tool/environment identifiers are retained here. No independent proof acceptance is claimed.
