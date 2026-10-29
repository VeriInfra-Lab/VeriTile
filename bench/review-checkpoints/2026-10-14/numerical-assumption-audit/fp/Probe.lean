import bench.tritonbench_g.seeded_dropout.FPEquiv
import VeriTile.Triton.Float.TypedScalarArithmetic
import VeriTile.Meta.StatementAudit
open VeriTile Triton FP
#print_fp_assumptions VeriTile.Bench.TritonBenchG.SeededDropout.Equivalence.seeded_dropout_equiv
#eval (ScalarArithmetic.entry .addCommute (by decide)).rule.contract.instanceKey
#eval (ScalarArithmetic.entry .mulCommute (by decide)).rule.contract.instanceKey
#eval (ScalarArithmetic.entry .divMulRcp (by decide)).rule.contract.instanceKey
#eval TypedScalarArithmetic.entry.rule.contract.instanceKey
#eval (ScalarArithmetic.report .addCommute (by decide)).report.configuration.compress
#eval (ScalarArithmetic.report .mulCommute (by decide)).report.configuration.compress
#eval (ScalarArithmetic.report .divMulRcp (by decide)).report.configuration.compress
