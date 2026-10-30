import bench.repairs.tritonbench_g.triton_matmul.Proofs.RawEpilogue

namespace VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.RawSource
open VeriTile.Triton FP.Structural FPValue
open KcacheCopyTriton.TypedCopy (masked_scatter_memory)
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false

def outputOffset {α : Type} (p : Parameters) (s : State α) (i : CIndex p) : Nat :=
  ((p.stride_cm : Int) * offsetM p s i.1 + (p.stride_cn : Int) * offsetN p s i.2.1).toNat

def outputValue {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Parameters) (st : Storage) (s : State α) (i : CIndex p) : α :=
  M.cast none .real (dtype st) (RawSource.accumulation M D p st s i)

def active {α : Type} (p : Parameters) (s : State α) (i : CIndex p) : Bool :=
  maskM p s i.1 && maskN p s i.2.1

def written {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Parameters) (st : Storage) (initial : State α) : State α :=
  (TileShape.allIndices [BM p,BN p]).foldl (fun t i => if RawSource.active p initial i then
    t.write p.C (RawSource.outputOffset p initial i) (stored st (RawSource.outputValue M D p st initial i))
    else t) initial

theorem epilogueMemory {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Parameters) (st : Storage) (initial current : State α)
    (ctx : Context p initial current) (ha : HasAcc p (RawSource.accumulation M D p st initial) current) :
    (epilogueExecution M D p st initial current ctx ha).val.mem =
      (RawSource.written M D p st initial).mem := by
  cases st <;> simp [epilogueExecution,RawSource.written,masked_scatter_memory,ctx.memory,
    RawSource.active,RawSource.outputOffset,RawSource.outputValue,stored,dtype,backend,
    FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]

/-- Successful evaluation and the full memory transformer of the pasted body.
The computed iteration count and storage-dependent output cast are retained. -/
theorem execution_memory {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Parameters) (st : Storage) (initial : State α) :
    ∃ final, FP.Structural.exec (backend M D) (Source.typed p st) initial = some final ∧
      final.mem = (RawSource.written M D p st initial).mem := by
  let start := preludeExecution M D p st initial
  obtain ⟨mid,hm,hc,ha⟩ := loopExecution M D p st initial start.val
    (preludeContext M D p st initial).1 (preludeContext M D p st initial).2
  let finish := epilogueExecution M D p st initial mid hc ha
  refine ⟨finish.val,?_,epilogueMemory M D p st initial mid hc ha⟩
  change run (backend M D) (Source.typed p st).surfaceBody initial = some finish.val
  rw [decomposition,run_append,run_append,start.property]
  simp only [Option.bind_some]
  have hb : start.val.regs .int [] "blocks_k" = some (fun _ => signedCeil p.K (BK p)) :=
    (preludeContext M D p st initial).1.blocks
  have hr : run (backend M D)
      [.forRangeInt "k" (.constInt 0) (.ref .int [] "blocks_k") (.constNat 1) (body p st)]
      start.val = some mid := by
    simpa [run,step,evalExpr,evalOp_unfold,hb] using hm
  rw [hr]
  exact finish.property

#axiomsClean epilogueMemory
#axiomsClean execution_memory
end
end VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.RawSource
