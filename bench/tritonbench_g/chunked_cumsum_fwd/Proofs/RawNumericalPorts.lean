import bench.tritonbench_g.chunked_cumsum_fwd.Proofs.RawObserved
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd
noncomputable section
open scoped Classical
#check VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd.RawObserved.NativeModel.add
#check VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd.RawObserved.NativeModel.cast
#check VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd.RawObserved.NativeModel.exp
#check VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd.RawObserved.NativeModel.fma
#check VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd.RawObserved.NativeModel.lessEqual
#check VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd.RawObserved.NativeModel.literal
#check VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd.RawObserved.NativeModel.log
#check VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd.RawObserved.NativeModel.maximum
#check VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd.RawObserved.NativeModel.minimum
#check VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd.RawObserved.NativeModel.mul
#check fun (M : RawObserved.NativeModel (BitVec 32)) (native_literal_fp32 : ℝ → BitVec 32) => (∀ x ∈ ({0,1,20} : Finset ℝ), M.literal x = native_literal_fp32 x)
#check fun (M : RawObserved.NativeModel (BitVec 32)) (native_cast_fp32 : FloatDType → FloatDType → BitVec 32 → BitVec 32) => (∀ src dst : FloatDType, (src,dst) ∈ [(.fp32,.real),(.real,.real),(.real,.fp32)] → ∀ x : BitVec 32, M.cast src dst x = native_cast_fp32 src dst x)
#check fun (M : RawObserved.NativeModel (BitVec 32)) (native_add_fp32 : BitVec 32 → BitVec 32 → BitVec 32) => (∀ x y : BitVec 32, M.add x y = native_add_fp32 x y)
#check fun (M : RawObserved.NativeModel (BitVec 32)) (native_mul_fp32 : BitVec 32 → BitVec 32 → BitVec 32) => (∀ x y : BitVec 32, M.mul x y = native_mul_fp32 x y)
#check fun (M : RawObserved.NativeModel (BitVec 32)) (native_maximum_fp32 : BitVec 32 → BitVec 32 → BitVec 32) => (∀ x y : BitVec 32, M.maximum x y = native_maximum_fp32 x y)
#check fun (M : RawObserved.NativeModel (BitVec 32)) (native_minimum_fp32 : BitVec 32 → BitVec 32 → BitVec 32) => (∀ x y : BitVec 32, M.minimum x y = native_minimum_fp32 x y)
#check fun (M : RawObserved.NativeModel (BitVec 32)) (native_exp_fp32 : BitVec 32 → BitVec 32) => (∀ x : BitVec 32, M.exp x = native_exp_fp32 x)
#check fun (M : RawObserved.NativeModel (BitVec 32)) (native_log_fp32 : BitVec 32 → BitVec 32) => (∀ x : BitVec 32, M.log x = native_log_fp32 x)
#check fun (M : RawObserved.NativeModel (BitVec 32)) (native_le_fp32 : BitVec 32 → BitVec 32 → Bool) => (∀ x y : BitVec 32, M.lessEqual x y = native_le_fp32 x y)
#check fun (M : RawObserved.NativeModel (BitVec 32)) (native_fma_rn_fp32 : BitVec 32 → BitVec 32 → BitVec 32 → BitVec 32) => (∀ x y z : BitVec 32, M.fma x y z = native_fma_rn_fp32 x y z)
#guard_msgs (drop info) in
#auditModuleAxioms
end
