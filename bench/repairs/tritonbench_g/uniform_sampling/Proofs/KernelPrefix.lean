import bench.repairs.tritonbench_g.uniform_sampling.Proofs.Projection
import bench.tritonbench_g.uniform_sampling.Proofs.TypedBindings
import bench.tritonbench_g.uniform_sampling.Proofs.TypedTuple

/-! The complete repaired prefix, including opaque scalar ABI words, signed64
index widening, modular low-word addition and all four canonical Philox words. -/
namespace VeriTile.Bench.Repairs.UniformSampling.KernelPrefix
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.UniformSampling
open TypedBindings
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 20000

def scalarStmt (name : RegName) (word : Float32Bits) : ComputeStmt :=
  .assign .real [] name (.compute (.bitcast .uint32 .fp32 rfl
    (.alg .uint32 (.constNat word.bits.toNat))))

def scalarBody (lower upper : Float32Bits) : List ComputeStmt :=
  [scalarStmt "lower_arg" lower, scalarStmt "upper_arg" upper]

def scalarState {α : Type} (M : Algebra α) (lower upper : Float32Bits) (s : State α) : State α :=
  (s.setReg "lower_arg" .real [] (fun _ => M.fp32Bits lower)).setReg
    "upper_arg" .real [] (fun _ => M.fp32Bits upper)

theorem scalar_exec {α : Type} [Inhabited α] (M : Algebra α)
    (lower upper : Float32Bits) (s : State α) :
    run M (scalarBody lower upper) s = some (scalarState M lower upper s) := by
  cases lower
  cases upper
  simp [scalarBody,scalarStmt,scalarState,run,step,evalExpr,evalComputeOp,evalOp_unfold,
    Option.bind,BitVec.ofNat_toNat]

def indices {α : Type} (B : Nat) (s : State α) : TileIndex [B] → Nat :=
  fun i => s.pids 0 * B + i.1.val

def widened {α : Type} (B : Nat) (s : State α) : State α :=
  (s.setReg "pid" .int [] (fun _ => (s.pids 0 : Int))).setReg
    "lane" .int [B] (fun i => (i.1.val : Int))

def indexState {α : Type} (B : Nat) (s : State α) : State α :=
  (widened B s).setReg "i4" .int [B] (fun i => (indices B s i : Int))

def wrappedState {α : Type} (offset : Int) (B : Nat) (s : State α) : State α :=
  (indexState B s).setReg "c0" .nat [B]
    (fun i => (ScalarSpec.counterLow offset (indices B s i)).toNat)

def preparedState {α : Type} (offset : Int) (B : Nat) (s : State α) : State α :=
  (wrappedState offset B s).setReg "_O" .nat [B] (fun _ => 0)

def wrappedOp (B : Nat) : Op .nat [B] :=
  .intToUnsigned 32 (.add .int .scalarL (.castNatToInt (.ref .nat [] "c0"))
    (.castNatToInt (.intToUnsigned 32 (.ref .int [B] "i4"))))

def indexBody (swapped : Bool) (B : Nat) : List ComputeStmt := [
  .assign .int [] "pid" (.alg (.castNatToInt (.programId 0))),
  .assign .int [B] "lane" (.alg (.castNatToInt (.arange B))),
  KernelSource.indexStmt swapped B,
  .assign .nat [B] "c0" (.alg (wrappedOp B)),
  .assign .nat [B] "_O" (.alg (.intToUnsigned 32 (.mul .int .scalarR
    (.castNatToInt (.ref .nat [B] "c0")) (.constInt 0))))]

theorem wrapped_add (offset : Int) (index : Nat) :
    (((((BitVec.ofInt 32 offset).toNat : Int) +
      (((index : Int) % (2^32 : Int)).toNat : Int)) % (2^32 : Int)).toNat) =
      (ScalarSpec.counterLow offset index).toNat := by
  rw [Int.toNat_of_nonneg (Int.emod_nonneg _ (by norm_num : (2^32 : Int) ≠ 0))]
  rw [Int.add_emod, Int.emod_emod, ← Int.add_emod]
  exact CounterOperands.wrapped_low_add offset index

theorem index_exec {α : Type} [Inhabited α] (M : Algebra α)
    (swapped : Bool) (offset : Int) (B : Nat) (s : State α)
    (hb : s.regs .nat [] "BLOCK_arg" = some (fun _ => B))
    (hlo : s.regs .nat [] "c0" = some (fun _ => (BitVec.ofInt 32 offset).toNat)) :
    run M (indexBody swapped B) s = some (preparedState offset B s) := by
  have hi : step M (KernelSource.indexStmt swapped B) (widened B s) =
      some (indexState B s) := by
    cases swapped <;>
      simp [KernelSource.indexStmt,KernelSource.indexOp,step,evalExpr,evalOp_unfold,
        widened,indexState,indices,hb,numeric,Nat.cast_add,Nat.cast_mul]
    all_goals congr 1
    all_goals funext i
    all_goals simp [bop,Int.mul_comm]
  have hw : evalOp M none (wrappedOp B) (indexState B s) =
      some (fun i => (ScalarSpec.counterLow offset (indices B s i)).toNat) := by
    have hc : (indexState B s).regs .nat [] "c0" =
        some (fun _ => (BitVec.ofInt 32 offset).toNat) := by simpa [indexState,widened] using hlo
    have hi4 : (indexState B s).regs .int [B] "i4" = some (fun i => (indices B s i : Int)) := by
      simp [indexState]
    simp [wrappedOp,evalOp_unfold,hc,hi4,Option.bind,bop,numeric,integralMod,IntegerExpr.sameShape]
    funext i
    simpa [indices,BitVec.toNat_ofInt] using wrapped_add offset (indices B s i)
  have hz : evalOp M none (.intToUnsigned 32 (.mul .int .scalarR
      (.castNatToInt (.ref .nat [B] "c0")) (.constInt 0)))
      (wrappedState offset B s) = some (fun _ => 0) := by
    simp [evalOp_unfold,wrappedState,bop,numeric,integralMod]
  change run M ((indexBody swapped B).take 2 ++ (indexBody swapped B).drop 2) s = _
  rw [run_append]
  have hstart : run M ((indexBody swapped B).take 2) s = some (widened B s) := by
    simp [indexBody,run,step,evalExpr,evalOp_unfold,widened]
  rw [hstart,Option.bind_some]
  change run M [KernelSource.indexStmt swapped B,
    .assign .nat [B] "c0" (.alg (wrappedOp B)),
    .assign .nat [B] "_O" (.alg (.intToUnsigned 32 (.mul .int .scalarR
      (.castNatToInt (.ref .nat [B] "c0")) (.constInt 0))))] (widened B s) = _
  exact three_steps M hi (assign_step_of_eval M "c0" _ _ _ hw)
    (assign_step_of_eval M "_O" _ _ _ hz)

def beforeTuple {α : Type} (M : Algebra α) (seed offset : Int)
    (lower upper : Float32Bits) (B : Nat) (s : State α) : State α :=
  preparedState offset B (counterState seed offset B (scalarState M lower upper s))

def result {α : Type} (M : Algebra α) (seed offset : Int)
    (lower upper : Float32Bits) (B : Nat) (s : State α) : State α :=
  TypedTuple.outputState seed offset (indices B s)
    (TypedTuple.capturedState seed offset (indices B s) (beforeTuple M seed offset lower upper B s))

theorem source_split (swapped : Bool) (y : RegionName) (N : Nat) (seed offset : Int)
    (lower upper : Float32Bits) (B : Nat) (hB : Random.ArangeSizeDomain B) :
    (KernelSource.source swapped y N seed offset lower upper B hB).surfaceBody.take 15 =
      ((scalarBody lower upper ++ TypedBindings.body seed offset B) ++ indexBody swapped B) ++
        [TypedTuple.tupleBlock B] := by
  cases swapped <;> rfl

theorem prefix_exec {α : Type} [Inhabited α] (M : Algebra α)
    (swapped : Bool) (y : RegionName) (N : Nat) (seed offset : Int)
    (lower upper : Float32Bits) (B : Nat) (hB : Random.ArangeSizeDomain B) (s : State α)
    (hseed : Random.IntegerType.i64.inRange seed = Bool.true) :
    run M ((KernelSource.source swapped y N seed offset lower upper B hB).surfaceBody.take 15) s =
      some (result M seed offset lower upper B s) := by
  rw [source_split,run_append,run_append,run_append,scalar_exec,Option.bind_some,
    TypedBindings.prefix_exec,Option.bind_some,index_exec M swapped offset B _
      (by simp [counterState,lowState,boundState]) (by simp [counterState,lowState]),Option.bind_some]
  simp only [run]
  rw [TypedTuple.tuple_exec M seed offset (indices B s) _ hseed]
  · rfl
  all_goals simp [indices,preparedState,wrappedState,indexState,widened,
    counterState,lowState,boundState,scalarState]

#axiomsClean scalar_exec
#axiomsClean wrapped_add
#axiomsClean index_exec
#axiomsClean source_split
#axiomsClean prefix_exec
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.UniformSampling.KernelPrefix
