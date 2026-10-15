# Official public-entrypoint replay at 93d38e39

The unchanged official comparator accepted **6/6 files and 26/26 theorem
targets**, covering all **11 registered public declarations** in the first
scope-extension batch at commit `93d38e399ac513eb98aa3624d9aeb1c21858757a`.
Every official runner and comparator returned exit 0. The 26 targets include
generated/private helpers; they are not 26 public declarations or completed
programs. [verification.json](verification.json) and
[original-results.json](original-results.json) preserve the results;
[target-mapping.json](target-mapping.json) preserves the exact mapping.

This is historical same-source replay: challenge and solution use the same
generated module, as the unchanged repository runner specifies. Acceptance is
by the Lean default kernel with the three standard permitted axioms;
`enable_nanoda=false`. It is not comparison against a separately supplied
original-program specification. Exact theorem premises, stage/dtype limits,
callbacks and native gaps remain unchanged. **Whole-program acceptance stays
0/183.** No current-source or later-batch result is inferred.

| Frozen file | Actual theorem targets | Registered declarations |
|---|---:|---:|
| attention `FullCorrect` | 2 | 2 |
| FP4 `HostCorrect` | 6 | 1 |
| kcache `DiscreteCorrect` | 4 | 1 |
| kcache `SignedCorrect` | 8 | 1 |
| multinomial `HostCorrect` | 4 | 4 |
| convolution `BranchesCorrect` | 2 | 2 |

The supported isolated VM used nonprivileged UID 2000, read-only source/tool
media, Landlock ABI 6 and systemd AF_UNIX restrictions. Original positive and
negative controls verify acceptance of a valid proof and rejection of changed
statements, `sorryAx` and source writes. Full logs/configurations/inventories,
source guards, original scripts and tool/isolation provenance are retained in
[replay-evidence.tar.gz](replay-evidence.tar.gz). Its SHA256 is
`1ba741709746f29d0499d1560820f1bece91b99e70bde5934419c2db6bc67150`.

[independent-review.json](independent-review.json) records independent checking
of all 7,177 historical Git blobs, all 4,303 guarded official inputs, six
reconstructed source wrappers and all eleven statement hashes. Root also
checked the archive members and every retained payload byte/hash. All 99 files
are ordinary files, with no unsafe paths or links. [retention.json](retention.json)
binds the compact copies and limits. The separate earlier
[118-target library replay](../comparator-original-scope-library/README.md)
is not included in these counts and was not rerun during evidence review.

Inspect the complete evidence in a fresh temporary directory:

```sh
evidence_dir=$(mktemp -d /tmp/veritile-public-evidence.XXXXXX)
tar -xzf replay-evidence.tar.gz -C "$evidence_dir"
cd "$evidence_dir/public-entrypoints-93d38e39-evidence"
sha256sum --check SHA256SUMS
```

The compressed package excludes build caches, proof/tool binaries, VM disks,
and credentials. No current catalog, source proof or acceptance ledger is
changed by retaining this historical evidence.
