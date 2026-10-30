import bench.tritonbench_g.layer_norm_liger.IO
import bench.tritonbench_g.layer_norm_liger.Proofs.SourceSafety
import bench.tritonbench_g.layer_norm_liger.Proofs.Frame

namespace VeriTile.Bench.Repairs.LayerNormLiger.LayerIO
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.TokenAttnReduceV.FPStages (SumPrimitive withSum)
noncomputable section
variable {α : Type} [Inhabited α]
set_option linter.unusedSimpArgs false

@[simp] theorem f_pid (M : Algebra α) (R : SumPrimitive α) (p : Forward.Parameters)
    (pid : Nat) (s : State α) : ((fInvocation M R p pid).initial s).pids 0=pid := by
  simp [fInvocation,Invocation.initial,bindRegisters]
@[simp] theorem b_pid (M : Algebra α) (R : SumPrimitive α) (p : Backward.Parameters)
    (pid : Nat) (s : State α) : ((bInvocation M R p pid).initial s).pids 0=pid := by
  simp [bInvocation,Invocation.initial,bindRegisters]

theorem forward_safe (M : Algebra α) (R : SumPrimitive α) (p : Forward.Parameters)
    (pid : Nat) (v : ChannelValues α (fInputs p)) (bounds : RegionName → Nat)
    (hb : (forwardIO M R p pid).InBounds bounds v) (s : State α) :
    ScheduleSafe bounds ((forwardIO M R p pid).schedule v) s := by
  change ScheduleSafe bounds [fInvocation M R p pid] s
  rw [scheduleSafe_singleton]
  apply SourceSafety.forward
  · intro i hi
    simpa [forwardIO,fMemory,fInputs] using hb.2.1 (0 : Fin 3) (i,PUnit.unit) hi
  · intro i hi
    exact hb.2.1 (1 : Fin 3) (i,PUnit.unit) hi
  · intro i hi
    exact hb.2.1 (2 : Fin 3) (i,PUnit.unit) hi
  · simpa [forwardIO,fMemory,fOutputs] using hb.2.2.1 (1 : Fin 3) PUnit.unit trivial
  · simpa [forwardIO,fMemory,fOutputs] using hb.2.2.1 (2 : Fin 3) PUnit.unit trivial
  · intro i hi
    simpa [forwardIO,fMemory,fOutputs] using hb.2.2.1 (0 : Fin 3) (i,PUnit.unit) hi

theorem backward_bounds (M : Algebra α) (R : SumPrimitive α) (p : Backward.Parameters)
    (pid : Nat) (v : ChannelValues α (bInputs p)) (bounds : RegionName → Nat)
    (hb : (backwardIO M R p pid).InBounds bounds v) : SourceSafety.Backward.Bounds p pid bounds := by
  constructor
  · intro row lo hi i hc
    have hr : row<p.rows := lt_of_lt_of_le hi (Nat.min_le_right _ _)
    exact hb.2.1 (0 : Fin 5) (⟨row,hr⟩,i,PUnit.unit) ⟨⟨lo,hi⟩,hc⟩
  · intro hi i hc
    exact hb.2.1 (1 : Fin 5) (i,PUnit.unit) ⟨hi,hc⟩
  · intro row lo hi i hc
    have hr : row<p.rows := lt_of_lt_of_le hi (Nat.min_le_right _ _)
    exact hb.2.1 (2 : Fin 5) (⟨row,hr⟩,i,PUnit.unit) ⟨⟨lo,hi⟩,hc⟩
  · intro row lo hi
    have hr : row<p.rows := lt_of_lt_of_le hi (Nat.min_le_right _ _)
    exact hb.2.1 (3 : Fin 5) (⟨row,hr⟩,PUnit.unit) ⟨lo,hi⟩
  · intro row lo hi
    have hr : row<p.rows := lt_of_lt_of_le hi (Nat.min_le_right _ _)
    exact hb.2.1 (4 : Fin 5) (⟨row,hr⟩,PUnit.unit) ⟨lo,hi⟩
  · intro row lo hi i hc
    have hr : row<p.rows := lt_of_lt_of_le hi (Nat.min_le_right _ _)
    exact hb.2.2.1 (0 : Fin 3) (⟨row,hr⟩,i,PUnit.unit) ⟨⟨lo,hi⟩,hc⟩
  · intro i hc
    exact hb.2.2.1 (1 : Fin 3) (i,PUnit.unit) hc
  · intro i hc
    exact hb.2.2.1 (2 : Fin 3) (i,PUnit.unit) hc

theorem backward_safe (M : Algebra α) (R : SumPrimitive α) (p : Backward.Parameters)
    (pid : Nat) (v : ChannelValues α (bInputs p)) (bounds : RegionName → Nat)
    (hb : (backwardIO M R p pid).InBounds bounds v) (s : State α) :
    ScheduleSafe bounds ((backwardIO M R p pid).schedule v) s := by
  change ScheduleSafe bounds [bInvocation M R p pid] s
  rw [scheduleSafe_singleton]
  apply SourceSafety.Backward.source_safe
  simpa only [b_pid] using backward_bounds M R p pid v bounds hb

theorem forward_outside (M : Algebra α) (R : SumPrimitive α) (p : Forward.Parameters)
    (pid : Nat) (v : ChannelValues α (fInputs p)) (s : State α) (r : RegionName) (o : Nat)
    (h : (forwardIO M R p pid).Outside v r o) :
    Forward.Outside p ((fInvocation M R p pid).initial s) r o := by
  refine ⟨?_,?_,?_⟩
  · simpa [forwardIO,Forward.meanOffset,fMemory,fOutputs] using h.1 (1 : Fin 3) PUnit.unit trivial
  · simpa [forwardIO,Forward.rstdOffset,fMemory,fOutputs] using h.1 (2 : Fin 3) PUnit.unit trivial
  · by_cases hr : r=p.y
    · right
      intro i hi
      have cell := h.1 (0 : Fin 3) i hi
      simpa [forwardIO,hr,Forward.outputOffset,fMemory,fOutputs] using cell
    · exact Or.inl hr

theorem backward_outside (M : Algebra α) (R : SumPrimitive α) (p : Backward.Parameters)
    (pid : Nat) (v : ChannelValues α (bInputs p)) (s : State α) (r : RegionName) (o : Nat)
    (h : (backwardIO M R p pid).Outside v r o) :
    Backward.Outside p ((bInvocation M R p pid).initial s) r o := by
  refine ⟨?_,?_,?_⟩
  · by_cases hr : r=p.dw
    · right
      intro i hi
      simpa [backwardIO,hr,bMemory,bOutputs] using h.1 (1 : Fin 3) i hi
    · exact Or.inl hr
  · by_cases hr : r=p.db
    · right
      intro i hi
      simpa [backwardIO,hr,bMemory,bOutputs] using h.1 (2 : Fin 3) i hi
    · exact Or.inl hr
  · by_cases hr : r=p.dx
    · right
      intro row lo hi i hc
      have lo' : first p pid≤row := by simpa [Backward.start,first] using lo
      have hi' : row<last p pid := by simpa [Backward.stop,last] using hi
      have bound : row<p.rows := lt_of_lt_of_le hi' (Nat.min_le_right _ _)
      simpa [backwardIO,hr,bMemory,bOutputs] using h.1 (0 : Fin 3) (⟨row,bound⟩,i) ⟨⟨lo',hi'⟩,hc⟩
    · exact Or.inl hr

end
end VeriTile.Bench.Repairs.LayerNormLiger.LayerIO
