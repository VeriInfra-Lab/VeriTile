import bench.repairs.tritonbench_g.fast_rms_layernorm.Proofs.RealWeightTrace
import VeriTile.Triton.Memory.Denotation

namespace VeriTile.Bench.Repairs.FastRmsLayernorm.RealWeightFlat
open VeriTile.Triton
noncomputable section

def Outside (h : Call) (A : FlatAlloc) (pid K : Nat) (q : RegionName) (o : Nat) : Prop :=
  q≠A.flat ∨ ∀ j : Fin K,pid*K+j.val<h.cols → o≠A.addr "dw" (pid*K+j.val)

/-- The new source executes on finite, disjoint flat storage. Bounds come from
the successful wrapper allocation, rather than from an execution premise. -/
theorem complete (h : Call) (K : Nat) (A : FlatAlloc) (hd : A.Disjoint)
    (closed : ∀ q,q∉A.regions → A.extent q=0) (u : BlockState)
    (undef : u.undef=fun _ _ => 0) (bounds : RealWeightSafety.Bounds h K (u.pids 0) A.extent) :
    ∃ v,exec (A.flattenKernel (RealWeightProgram.source h K)) (A.flattenState u)=some v ∧
      (∀ j : Fin K,u.pids 0*K+j.val<h.cols →
        v.readMem A.flat (A.addr "dw" (u.pids 0*K+j.val))=
          RealWeightLoop.carry h K (u.pids 0) u h.rows j) ∧
      (∀ q o,Outside h A (u.pids 0) K q o → v.mem q o=(A.flattenState u).mem q o) := by
  obtain ⟨v,he,ho,hf⟩:=RealWeightMemory.complete h K u
  have bridge:=A.exec_flatten hd closed _ u (RealWeightTrace.complete h K A.extent u bounds)
    (RealWeightTrace.flatten_ok h K) undef
  refine ⟨A.flattenState v,by rw [bridge,he,Option.map_some],?_,?_⟩
  · intro j hj
    have member : ("dw":RegionName)∈A.regions := by
      by_contra outside
      have := bounds.output j hj
      rw [closed "dw" outside] at this
      omega
    rw [A.flattenState_readMem hd v member (bounds.output j hj)]
    exact ho j hj
  · intro q o outside
    by_cases eq : q=A.flat
    · subst q
      simp only [FlatAlloc.flattenState]
      unfold FlatAlloc.readFlat
      cases dec : A.decode o with
      | none => rfl
      | some pair =>
        obtain ⟨r,k⟩:=pair
        obtain ⟨_,addr,_⟩:=A.decode_sound dec
        apply congrArg A.trCell (hf r k ?_)
        rintro ⟨rfl,j,hj,rfl⟩
        exact outside.elim (fun ne => ne rfl) (fun out => out j hj addr)
    · simp only [FlatAlloc.flattenState,if_neg eq]

#axiomsClean complete
end
end VeriTile.Bench.Repairs.FastRmsLayernorm.RealWeightFlat
