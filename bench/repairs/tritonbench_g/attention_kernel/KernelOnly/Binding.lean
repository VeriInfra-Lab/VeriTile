import bench.repairs.tritonbench_g.attention_kernel.KernelOnly.Specification

namespace VeriTile.Bench.Repairs.AttentionKernel.KernelOnly
open VeriTile Triton FP.Structural Whole001 Source Whole Machine Whole002
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem query_check (p : Parameters) (s : State α) (i : TileIndex [p.bm,p.dim]) :
    (qptr p s).inBounds (TileShape.indexToList _ i) [0]=decide (s.pids 0*p.bm+i.1.val<p.length) := by
  change (decide (s.pids 0*p.bm+i.1.val<p.length) && Bool.true)=_
  simp
theorem key_check (p : Parameters) (s : State α) (row : Nat) (i : TileIndex [p.dim,p.bn]) :
    (kptr p s row).inBounds (TileShape.indexToList _ i) [1]=decide (row+i.2.1.val<p.length) := by
  change (decide (row+i.2.1.val<p.length) && Bool.true)=_
  simp
theorem value_check (p : Parameters) (s : State α) (row : Nat) (i : TileIndex [p.bn,p.dim]) :
    (vptr p s row).inBounds (TileShape.indexToList _ i) [0]=decide (row+i.1.val<p.length) := by
  change (decide (row+i.1.val<p.length) && Bool.true)=_
  simp

/-- Operand-by-operand connection to the independent recurrence. The only
premises here are initial reads and integer PID coordinates, never a result. -/
theorem source_algorithm (M : Algebra α) (dot : Dot α) (p : Parameters)
    (s : State α) (read : RegionName → Nat → α) (block batch head : Nat)
    (hblock : s.pids 0=block) (hbatch : s.pids 1/p.heads=batch) (hhead : s.pids 1%p.heads=head)
    (loads : ∀ r off,toFloat p.storage.float ((s.mem r off).read p.storage.tile)=read r off) :
    answer M dot p s=Specification.algorithm M dot p read block batch head := by
  have q : Source.query M p s=Specification.query M p read block batch head := by
    funext i
    simp [Source.query,Specification.query,query_check,loads,Source.base,hblock,hbatch,hhead,offset]
  have k : ∀ row,Source.key M p s row=Specification.key M p read block batch head row := by
    intro row
    funext i
    simp [Source.key,Specification.key,key_check,loads,Source.base,hblock,hbatch,hhead,offset,
      Nat.add_assoc,Nat.add_left_comm,Nat.add_comm]
  have v : ∀ row,Source.value M p s row=Specification.value M p read block batch head row := by
    intro row
    funext i
    simp [Source.value,Specification.value,value_check,loads,Source.base,hblock,hbatch,hhead,offset]
  have b1 : Source.biasStatic M p s=Specification.biasStatic M p read block batch head := by
    funext i
    simp [Source.biasStatic,Specification.biasStatic,loads,Source.base,hblock,hbatch,hhead]
  have b0 : ∀ row,Source.biasDynamic M p s row=Specification.biasDynamic M p read block batch head row := by
    intro row
    funext i
    simp [Source.biasDynamic,Specification.biasDynamic,loads,Source.base,hblock,hbatch,hhead]
  have scores : Source.scores M dot p s=Specification.scores M dot p read block batch head := by
    funext row i
    simp only [Source.scores,Specification.scores,q,k,b0,b1]
    rfl
  have maximum : Source.maximum M dot p s=Specification.maximum M dot p read block batch head := by
    funext row acc i
    simp only [Source.maximum,Specification.maximum,scores]
  have probability : Source.probability M dot p s=Specification.probability M dot p read block batch head := by
    funext row acc i
    simp only [Source.probability,Specification.probability,scores,maximum]
  have rescale : Source.rescale M dot p s=Specification.rescale M dot p read block batch head := by
    funext row acc i
    simp only [Source.rescale,Specification.rescale,maximum]
  have contribution : Source.contribution M dot p s=Specification.contribution M dot p read block batch head := by
    funext row acc i
    simp only [Source.contribution,Specification.contribution,probability,v]
  have step : Source.update M dot p s=Specification.update M dot p read block batch head := by
    funext row acc
    simp only [Source.update,Specification.update,maximum,rescale,probability,contribution]
  unfold Whole.answer Source.outputValue Source.accumulated Specification.algorithm
  rw [step]
  rfl

theorem entry_reads (c : Call) (read : RegionName → Nat → α) (g : Whole.Grid) (pid : Program)
    (r : RegionName) (off : Nat) :
    toFloat (parameters c).storage.float
      (((start g pid (entry c read)).mem r off).read (parameters c).storage.tile)=read r off := by
  simp [start,entry,parameters,Real.float_roundtrip]

theorem algorithm_at (M : Algebra α) (dot : Dot α) (c : Call) (read : RegionName → Nat → α)
    (b h n : Nat) (hh : h<(parameters c).heads) :
    answer M dot (parameters c)
      (start (grid (parameters c)) (n/c.bm,b*(parameters c).heads+h,0) (entry c read))=
      Specification.algorithm M dot (parameters c) read (n/c.bm) b h := by
  have positive : 0<(parameters c).heads := by omega
  have div : (b*(parameters c).heads+h)/(parameters c).heads=b := by
    rw [Nat.add_comm,Nat.add_mul_div_right _ _ positive,Nat.div_eq_of_lt hh]
    omega
  have mod : (b*(parameters c).heads+h)%(parameters c).heads=h := by
    simp [Nat.add_mod,Nat.mod_eq_of_lt hh]
  apply source_algorithm
  · simp [start,ids]
  · simpa [start,ids] using div
  · simpa [start,ids] using mod
  · exact entry_reads c read _ _

#axiomsClean source_algorithm
#axiomsClean entry_reads
#axiomsClean algorithm_at
end
end VeriTile.Bench.Repairs.AttentionKernel.KernelOnly
