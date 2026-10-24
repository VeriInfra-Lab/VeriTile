# Final corpus source audit after the signed-cast review

The corrected-PATH complete `bench/audit_tritonbench_g.sh --source-only` run passed all source/placeholder/manifest/registry/statement/mask/control gates for 178 extracted views. All 3,595 captured inputs were unchanged. This receipt contains only the complete corpus source-only run; the 49 materialization regression tests and their versioned fixtures are in `../multinomial-cast-source-regressions`.

The current inventory passes its separate `--check` run: 184 task directories, 178 source pairs, 697 public declarations in 289 files, no metadata errors. It retains 58 full-value labels requiring acceptance, 103 specialized, 16 partial, 1 blocked and 6 missing-source tasks. No task's scope label was promoted by this frontend change. These gates are source/metadata checks, not Lean proof replay or independent acceptance.
