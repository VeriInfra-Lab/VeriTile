import bench.tritonbench_g.attention_fwd_triton3.Proofs.Reference

namespace VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.Reference
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.AttentionFwdTriton3.Host
open VeriTile.Bench.Repairs.ContextAttnMistral
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 40000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def finalCenter (c : Call) (o : Ops α) (old : Carry c α) : Values α .real [c.BM] :=
  fun i => if c.finish≠0 then o.scalar.binary (some .fp32) .real .add
    (old.center i) (o.scalar.unary (some .fp32) .log2 (old.mass i)) else old.center i

def finalOutput (c : Call) (o : Ops α) (old : Carry c α) : Values α .real [c.BM,c.D] :=
  fun i => o.scalar.cast (some .fp32) .real .real
    (if c.finish≠0 then o.scalar.binary (some .fp32) .real .div (old.weighted i) (old.mass (i.1,()))
      else old.weighted i)

def writeRows (c : Call) (a : Inputs α) (rg : RegionName) (v : Values α .real [c.BM]) (s : State α) : State α :=
  (TileShape.allIndices [c.BM]).foldl
    (fun t i => t.write rg (carryAddress c a i.1.val) (.mk .real (v i))) s

def writeOutput (c : Call) (a : Inputs α) (v : Values α .real [c.BM,c.D]) (s : State α) : State α :=
  (TileShape.allIndices [c.BM,c.D]).foldl
    (fun t i => t.write (optr c a).region ((optr c a).address (TileShape.indexToList [c.BM,c.D] i))
      (.mk .real (v i))) s

/-- Actual padded stores in source order: L only before END, then M, then O.
The arbitrary initial memory supplies every byte outside these writes. -/
def scatter (c : Call) (o : Ops α) (a : Inputs α) (old : Carry c α) (s : State α) : State α :=
  let denominator := if c.finish=0 then writeRows c a "L" old.mass s else s
  let center := writeRows c a "M" (finalCenter c o old) denominator
  writeOutput c a (finalOutput c o old) center

@[simp] theorem fold_write_regs {I : Type} (is : List I) (rg : RegionName)
    (off : I → Nat) (v : I → Cell α) (s : State α) :
    (is.foldl (fun t i => t.write rg (off i) (v i)) s).regs=s.regs := by
  induction is generalizing s with
  | nil => rfl
  | cons i rest ih => exact ih _

theorem fold_write_mem_congr {I : Type} (is : List I) (rg : RegionName)
    (off : I → Nat) (v : I → Cell α) (s t : State α) (memory : s.mem=t.mem) :
    (is.foldl (fun u i => u.write rg (off i) (v i)) s).mem=
      (is.foldl (fun u i => u.write rg (off i) (v i)) t).mem := by
  induction is generalizing s t with
  | nil => exact memory
  | cons i rest ih =>
    simp only [List.foldl_cons]
    exact ih _ _ (by simp only [State.write,memory])

theorem scatter_mem_congr (c : Call) (o : Ops α) (a : Inputs α) (old : Carry c α)
    (s t : State α) (memory : s.mem=t.mem) :
    (scatter c o a old s).mem=(scatter c o a old t).mem := by
  unfold scatter writeOutput
  apply fold_write_mem_congr
  unfold writeRows
  apply fold_write_mem_congr
  split_ifs
  · apply fold_write_mem_congr
    exact memory
  · exact memory

theorem epilogue (c : Call) (o : Ops α) (a : Inputs α) (old : Carry c α) (s : State α)
    (live : Matches c o a old s) :
    ∃ t,run (engine o.algebra) (parts c).after s=some t ∧
      t.mem=(scatter c o a old s).mem := by
  rcases live with ⟨h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12⟩
  by_cases hf : c.finish=0
  all_goals
    simp (config := {maxSteps := 500000}) [parts,Kernels.source,Quoted.kernel,
      ComputeKernel.surfaceBody,Source.intRange?,run,step,evalExpr,evalComputeOp,evalOp_unfold,
      Option.bind,ComputeDType.eraseDType,State.setReg,engine,Algebra.withDefaultPrecision,
      FPBackend.Ops.algebra,resolvePrecision,numeric,numericLt,numericLe,numericEq,
      natLt,natLe,integralFloorDiv,integralMod,ofFloat,toFloat,Region.cast,
      h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12,hf,
      store,scatter,writeRows,writeOutput,finalCenter,finalOutput,bop,
      Broadcast.leftIndex,Broadcast.rightIndex]
  all_goals apply fold_write_mem_congr
  all_goals apply fold_write_mem_congr
  all_goals rfl

#specNonCircular scatter avoiding [FP.Structural.exec,FP.Structural.run,Kernels.source,parts]
#axiomsClean epilogue
end
end VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.Reference
