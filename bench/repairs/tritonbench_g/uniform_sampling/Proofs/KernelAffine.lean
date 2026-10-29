import bench.repairs.tritonbench_g.uniform_sampling.Proofs.KernelPrefix
import bench.tritonbench_g.uniform_sampling.Proofs.TypedAffine

namespace VeriTile.Bench.Repairs.UniformSampling.KernelAffine
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.UniformSampling
open TupleExecution (fieldName)
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 20000

def expr (B : Nat) (name : RegName) : ComputeExpr .real [B] :=
  .fma .fp32 .scalarR .scalarR
    (.compute (.alg .fp32 (.uniform32 (.ref .nat [B] name))))
    (.compute (.alg .fp32 (.sub .real .nil
      (.ref .real [] "upper_arg") (.ref .real [] "lower_arg"))))
    (.compute (.alg .fp32 (.ref .real [] "lower_arg")))

def stmt (B : Nat) (name : RegName) : ComputeStmt := .assign .real [B] name (expr B name)

def values {α : Type} (M : Algebra α) (fused : α → α → α → α)
    {B : Nat} (words : TileIndex [B] → Random.Word) (lower upper : Float32Bits) : Values α .real [B] :=
  fun i => fused (M.fp32Bits ⟨(Random.uniform (words i)).bits⟩)
    (M.binary (some .fp32) .real .sub (M.fp32Bits upper) (M.fp32Bits lower)) (M.fp32Bits lower)

theorem eval {α : Type} [Inhabited α] (M : Algebra α) (fused : α → α → α → α)
    (hf : M.fma .fp32 = some fused) {B : Nat} (words : TileIndex [B] → Random.Word)
    (name : RegName) (lower upper : Float32Bits) (s : State α)
    (hw : s.regs .nat [B] name = some (fun i => (words i).toNat))
    (hl : s.regs .real [] "lower_arg" = some (fun _ => M.fp32Bits lower))
    (hu : s.regs .real [] "upper_arg" = some (fun _ => M.fp32Bits upper)) :
    evalExpr M (expr B name) s = some (values M fused words lower upper) := by
  have random := TypedRandom.eval_uniform words M (some .fp32) s (.ref .nat [B] name)
    (by simpa only [evalOp_unfold] using hw)
  simp only [expr,evalExpr,hf,bind,Option.bind,evalComputeOp]
  erw [random]
  simp [ComputeDType.eraseDType,evalOp_unfold,hl,hu,values,bop,numeric,Option.bind]
  rfl

theorem assign {α : Type} [Inhabited α] (M : Algebra α) (fused : α → α → α → α)
    (hf : M.fma .fp32 = some fused) {B : Nat} (words : TileIndex [B] → Random.Word)
    (name : RegName) (lower upper : Float32Bits) (s : State α)
    (hw : s.regs .nat [B] name = some (fun i => (words i).toNat))
    (hl : s.regs .real [] "lower_arg" = some (fun _ => M.fp32Bits lower))
    (hu : s.regs .real [] "upper_arg" = some (fun _ => M.fp32Bits upper)) :
    step M (stmt B name) s = some (s.setReg name .real [B] (values M fused words lower upper)) := by
  simp only [stmt,step,eval M fused hf words name lower upper s hw hl hu,bind,Option.bind,pure]

def body (B : Nat) : List ComputeStmt := [stmt B "r0",stmt B "r1",stmt B "r2",stmt B "r3"]

def result {α : Type} (M : Algebra α) (fused : α → α → α → α) {B : Nat}
    (words : Random.Direct.Field → TileIndex [B] → Random.Word)
    (lower upper : Float32Bits) (s : State α) : State α :=
  let s := s.setReg "r0" .real [B] (values M fused (words .c0) lower upper)
  let s := s.setReg "r1" .real [B] (values M fused (words .c1) lower upper)
  let s := s.setReg "r2" .real [B] (values M fused (words .c2) lower upper)
  s.setReg "r3" .real [B] (values M fused (words .c3) lower upper)

theorem exec {α : Type} [Inhabited α] (M : Algebra α) (fused : α → α → α → α)
    (hf : M.fma .fp32 = some fused) {B : Nat}
    (words : Random.Direct.Field → TileIndex [B] → Random.Word)
    (lower upper : Float32Bits) (s : State α)
    (hw : ∀ f, s.regs .nat [B] (fieldName f) = some (fun i => (words f i).toNat))
    (hl : s.regs .real [] "lower_arg" = some (fun _ => M.fp32Bits lower))
    (hu : s.regs .real [] "upper_arg" = some (fun _ => M.fp32Bits upper)) :
    run M (body B) s = some (result M fused words lower upper s) := by
  let s0 := s.setReg "r0" .real [B] (values M fused (words .c0) lower upper)
  let s1 := s0.setReg "r1" .real [B] (values M fused (words .c1) lower upper)
  let s2 := s1.setReg "r2" .real [B] (values M fused (words .c2) lower upper)
  apply TypedAffine.four_steps M
    (assign M fused hf (words .c0) "r0" lower upper s (hw .c0) hl hu)
    (assign M fused hf (words .c1) "r1" lower upper s0
      (by simpa [s0,fieldName] using hw .c1) (by simpa [s0] using hl) (by simpa [s0] using hu))
    (assign M fused hf (words .c2) "r2" lower upper s1
      (by simpa [s1,s0,fieldName] using hw .c2) (by simpa [s1,s0] using hl) (by simpa [s1,s0] using hu))
    (assign M fused hf (words .c3) "r3" lower upper s2
      (by simpa [s2,s1,s0,fieldName] using hw .c3) (by simpa [s2,s1,s0] using hl) (by simpa [s2,s1,s0] using hu))

theorem source_slice (swapped : Bool) (y : RegionName) (N : Nat) (seed offset : Int)
    (lower upper : Float32Bits) (B : Nat) (hB : Random.ArangeSizeDomain B) :
    ((KernelSource.source swapped y N seed offset lower upper B hB).surfaceBody.drop 15).take 4 = body B := by
  cases swapped <;> rfl

theorem result_register {α : Type} (M : Algebra α) (fused : α → α → α → α) {B : Nat}
    (words : Random.Direct.Field → TileIndex [B] → Random.Word)
    (lower upper : Float32Bits) (s : State α) (f : Random.Direct.Field) :
    (result M fused words lower upper s).regs .real [B] (fieldName f) =
      some (values M fused (words f) lower upper) := by
  cases f <;> simp [result,fieldName]

#axiomsClean eval
#axiomsClean exec
#axiomsClean source_slice
#axiomsClean result_register
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.UniformSampling.KernelAffine
