import bench.tritonbench_g.multinomial_sampling.Proofs.IOUndef
import bench.tritonbench_g.multinomial_sampling.IO

namespace VeriTile.Bench.Repairs.MultinomialSampling.IOFacts
open VeriTile.Triton FP.Structural SourceValue Interface
noncomputable section
variable {α : Type} [Inhabited α]
variable (rest : Algebra α) (P : Primitives α) (p : Parameters) (pid : Nat)
  (vals : (io rest P p pid).Inputs) (s : State α)
  (bound : (io rest P p pid).InputBinding vals s)

include bound

theorem score_read (r j : Nat) (hr : r<p.batches) (hj : j<p.tokens) :
    (s.mem p.scores (r*p.strideScoreBatch+j*p.strideScoreToken)).read .real =
      score p vals.memory r j := by
  have h := bound.1 (0 : Fin 4) (⟨r,hr⟩,⟨j,hj⟩,()) trivial
  simpa only [io,IOUndef.addUndefined,memoryIO,inputs,inputOffset,mapValue,id_eq,score,
    dif_pos (And.intro hr hj)] using h

theorem seed_read (r : Nat) (hr : r<p.batches) :
    (s.mem p.seeds r).read .int = seed p vals.memory r := by
  have h := bound.1 (1 : Fin 4) (⟨r,hr⟩,()) trivial
  simpa only [io,IOUndef.addUndefined,memoryIO,inputs,inputOffset,mapValue,id_eq,seed,dif_pos hr] using h

theorem offset_read (r : Nat) (hr : r<p.batches) :
    (s.mem p.offsets r).read .int = offset p vals.memory r := by
  have h := bound.1 (2 : Fin 4) (⟨r,hr⟩,()) trivial
  simpa only [io,IOUndef.addUndefined,memoryIO,inputs,inputOffset,mapValue,id_eq,offset,dif_pos hr] using h

theorem label_read (r j : Nat) (hr : r<p.batches) (hj : j<max p.tokens 1) :
    (s.mem p.indices (r*p.strideIndexBatch+j*p.strideIndexToken)).read .int =
      label p vals.memory r j := by
  have h := bound.1 (3 : Fin 4) (⟨r,hr⟩,⟨j,hj⟩,()) trivial
  simpa only [io,IOUndef.addUndefined,memoryIO,inputs,inputOffset,mapValue,id_eq,label,
    dif_pos (And.intro hr hj)] using h

theorem entry_binding :
    (io rest P p pid).InputBinding vals ((job rest P p pid).initial s) := by
  refine ⟨?_,?_⟩
  · intro i j active
    simpa only [Invocation.initial_mem] using bound.1 i j active
  · intro i j active
    simpa only [Invocation.initial_undef] using bound.2 i j active

omit bound in
@[simp] theorem entry_pid : ((job rest P p pid).initial s).pids 0=pid := rfl

theorem seed_value (samePid : s.pids 0=pid) (i : Row p) :
    SourceValue.seedValue p s i=Logical.seed p pid (data rest P p pid vals) i := by
  have h := IOUndef.seed_value p pid (memoryIO rest P p pid) vals s samePid bound.2
    (fun i _=>seed p vals.memory (IOUndef.row p pid i))
    (fun i active=>seed_read rest P p pid vals s bound _ active) i
  simpa only [IOUndef.seed,IOUndef.row,Logical.seed,Logical.row,data] using h

theorem offset_value (samePid : s.pids 0=pid) (i : Row p) :
    SourceValue.offsetValue p s i=Logical.offset p pid (data rest P p pid vals) i := by
  have h := IOUndef.offset_value p pid (memoryIO rest P p pid) vals s samePid bound.2
    (fun i _=>offset p vals.memory (IOUndef.row p pid i))
    (fun i active=>offset_read rest P p pid vals s bound _ active) i
  simpa only [IOUndef.offset,IOUndef.row,Logical.offset,Logical.row,data] using h

theorem initial_label (samePid : s.pids 0=pid) (i : Row p) :
    SourceValue.initialOutput p s i=Logical.initialLabel p pid (data rest P p pid vals) i := by
  have h := IOUndef.initial_label p pid (memoryIO rest P p pid) vals s samePid bound.2
    (fun i _=>label p vals.memory (IOUndef.row p pid i) 0)
    (fun i active=>by simpa using label_read rest P p pid vals s bound _ 0 active (by omega)) i
  simpa only [IOUndef.initialLabel,IOUndef.row,Logical.initialLabel,Logical.row,data] using h

/-- The logical signed seed condition is exactly the prior source condition. -/
theorem seed_domain_iff (samePid : s.pids 0=pid) :
    SourceValue.SeedDomain p s ↔ Interface.SeedDomain rest P p pid vals := by
  unfold SourceValue.SeedDomain Interface.SeedDomain Logical.SeedDomain
  simp_rw [seed_value rest P p pid vals s bound samePid]

#axiomsClean score_read
#axiomsClean seed_domain_iff
end
end VeriTile.Bench.Repairs.MultinomialSampling.IOFacts
