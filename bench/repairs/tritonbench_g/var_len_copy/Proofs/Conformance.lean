import bench.repairs.tritonbench_g.var_len_copy.Proofs.Native
import bench.repairs.tritonbench_g.var_len_copy.Proofs.Execution

namespace VeriTile.Bench.Repairs.VarLenCopy.Native
open VeriTile.Bench.TritonBenchG.VarLenCopy
noncomputable section
set_option maxHeartbeats 3000000

def sourceProgram (l : Layout) (pid : Nat) (s : Memory) :=
  execute (sourceEvents l (loadSegment l s pid)) s

def sourceLaunch (l : Layout) (s : Memory) :=
  (List.range l.count).foldl (fun t p=>sourceProgram l p t) s

theorem source_program (l : Layout) (r : Request) (s : Memory) (prep : Prepared l r s)
    (valid : Valid r) (p : Nat) (hp : p<r.count) :
    sourceProgram l p s=execute (events l (r.segment p)) s := by
  rw [sourceProgram,metadata_load l r s prep p hp,source_events l r s prep valid p hp]

theorem source_order (l : Layout) (r : Request) (s : Memory) (prep : Prepared l r s)
    (valid : Valid r) (order : List Nat) (bound : ∀ p∈order,p<r.count) :
    order.foldl (fun t p=>sourceProgram l p t) s=order.foldl (fun t p=>program l p t) s := by
  have relation := List.foldl_rel (l:=order) (a:=s) (b:=s)
    (f:=fun t p=>sourceProgram l p t) (g:=fun t p=>program l p t)
    (r:=fun t u=>t=u ∧ Prepared l r t) ⟨rfl,prep⟩ (by
      intro p hp t u same
      obtain ⟨rfl,prepared⟩ := same
      have active := bound p hp
      have hs := source_program l r t prepared valid p active
      have hg : program l p t=execute (events l (r.segment p)) t := by
        rw [program,metadata_load l r t prepared p active]
      refine ⟨hs.trans hg.symm,?_⟩
      change Prepared l r (sourceProgram l p t)
      rw [hs]
      exact prepared_execute l r t prepared valid p active)
  exact relation.1

theorem source_launch (l : Layout) (r : Request) (s : Memory)
    (prep : Prepared l r s) (valid : Valid r) : sourceLaunch l s=launch l s := by
  apply source_order l r s prep valid
  intro p hp
  simpa [prep.counts] using hp

#axiomsClean source_launch
end
end VeriTile.Bench.Repairs.VarLenCopy.Native
