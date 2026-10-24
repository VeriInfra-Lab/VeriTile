# Queue gzip integration scripts: documentation revision 2

This root-reviewed revision corrects one stale phrase in the proposed original scope document: “plain JSON numerical queue” becomes “compressed JSON numerical queue”. Counts, policy and all other document text are unchanged. The source-plan hash reflects those exact new document bytes.

The executable integration.py, both Python consumers, both regression test files and current_source_acceptance.md are byte-identical to reviewed revision 1. The full revision 1 package and its receipt closure are retained in evidence/previous-revision1; its root review and actual 16-test replay are in evidence/previous-revision1-root-review. Those results are inherited, not claimed as new executions. Runtime migration checks must still actually run. No new test was added for this wording change.

Root has reviewed this single-line delta. Follow PLAN.md using this package receipt hash and the actual clean post-four-task commit. This package performs no migration until its commands are run, and installs no primary gate or completion counts.
