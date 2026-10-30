import bench.repairs.tritonbench_g.fast_layernorm.Proofs.KernelAffine

namespace VeriTile.Bench.Repairs.FastLayernorm.KernelOutputs
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open KernelMemory
open KernelRows
noncomputable section
open scoped Classical
set_option maxHeartbeats 4000000
set_option maxRecDepth 20000
variable {α : Type} [Inhabited α]

def forwardFootprint (N B pid : Nat) (q : RegionName) (o : Nat) : Prop :=
  (q="y" ∧ ∃ j : Fin B,j.val<N ∧ o=pid*N+j.val) ∨ (q="r" ∧ o=pid) ∨ (q="mu" ∧ o=pid)
def backwardFootprint (N B pid : Nat) (q : RegionName) (o : Nat) : Prop :=
  q="dx" ∧ ∃ j : Fin B,j.val<N ∧ o=pid*N+j.val
def affineFootprint (wc bc pid : Nat) (bias : Bool) (q : RegionName) (o : Nat) : Prop :=
  (q="dw" ∧ ∃ j : Fin 128,pid*128+j.val<wc ∧ o=pid*128+j.val) ∨
  (bias=Bool.true ∧ q="db" ∧ ∃ j : Fin 128,pid*128+j.val<bc ∧ o=pid*128+j.val)

theorem forward_complete (M : Algebra α) (N B : Nat) (eps : ℝ) (bias : Bool) (u : State α) :
    ∃ v, FP.Structural.exec (engine M) (forwardKernel "y" "x" "w" "b" "r" "mu" N B eps bias) u=some v ∧
      v.mem=forwardHeap M N B eps bias u ∧
      (∀ j : Fin B,j.val<N → v.mem "y" (u.pids 0*N+j.val)=
        KernelReference.output M N eps bias (raw u "x" N B) (raw u "w" 0 B) (raw u "b" 0 B) j) ∧
      v.mem "r" (u.pids 0)=KernelReference.savedInverse M N eps (raw u "x" N B) ∧
      v.mem "mu" (u.pids 0)=KernelReference.savedMean M N (raw u "x" N B) ∧
      (∀ q o,¬forwardFootprint N B (u.pids 0) q o → v.mem q o=u.mem q o) := by
  obtain ⟨v,he,hm⟩ := forward_memory M N B eps bias u
  let middle := (u.write "r" (u.pids 0) (KernelReference.savedInverse M N eps (raw u "x" N B))).write
    "mu" (u.pids 0) (KernelReference.savedMean M N (raw u "x" N B))
  have memory : v.mem=commit "y" (fun i : TileIndex [B] => u.pids 0*N+i.1.val)
      (fun i => KernelReference.output M N eps bias (raw u "x" N B) (raw u "w" 0 B) (raw u "b" 0 B) i.1)
      (fun i => i.1.val<N) (TileShape.allIndices [B]) middle.mem := hm
  refine ⟨v,he,hm,?_,?_,?_,?_⟩
  · intro j hj
    rw [memory]
    exact commit_readback "y" (fun i : TileIndex [B] => u.pids 0*N+i.1.val)
      (fun i => KernelReference.output M N eps bias (raw u "x" N B) (raw u "w" 0 B) (raw u "b" 0 B) i.1)
      (fun i => i.1.val<N) middle (j,PUnit.unit) hj (row_injective _ _)
  · rw [memory,commit_frame _ _ _ _ _ middle _ _ (Or.inl (by decide))]
    simp [middle,State.write]
  · rw [memory,commit_frame _ _ _ _ _ middle _ _ (Or.inl (by decide))]
    exact State.write_same _ _ _ _
  · intro q o outside
    rw [memory,commit_frame]
    · change middle.mem q o = _
      dsimp only [middle]
      rw [State.write_other,State.write_other]
      · by_cases hq : q="r"
        · exact Or.inr (fun ho => outside (Or.inr (Or.inl ⟨hq,ho⟩)))
        · exact Or.inl hq
      · by_cases hq : q="mu"
        · exact Or.inr (fun ho => outside (Or.inr (Or.inr ⟨hq,ho⟩)))
        · exact Or.inl hq
    · by_cases hq : q="y"
      · exact Or.inr (fun i _ hi ho => outside (Or.inl ⟨hq,i.1,hi,ho⟩))
      · exact Or.inl hq

theorem backward_complete (M : Algebra α) (N B : Nat) (u : State α) :
    ∃ v, FP.Structural.exec (engine M) (backwardKernel "dx" "dy" "x" "w" "r" "mu" N B) u=some v ∧
      v.mem=backwardHeap M N B u ∧
      (∀ j : Fin B,j.val<N → v.mem "dx" (u.pids 0*N+j.val)=
        KernelReference.gradient M N (raw u "x" N B) (raw u "dy" N B) (raw u "w" 0 B)
          ((u.mem "mu" (u.pids 0)).read .real) ((u.mem "r" (u.pids 0)).read .real) j) ∧
      (∀ q o,¬backwardFootprint N B (u.pids 0) q o → v.mem q o=u.mem q o) := by
  obtain ⟨v,he,hm⟩ := backward_memory M N B u
  refine ⟨v,he,hm,?_,?_⟩
  · intro j hj
    rw [hm]
    exact commit_readback "dx" (fun i : TileIndex [B] => u.pids 0*N+i.1.val)
      (fun i => KernelReference.gradient M N (raw u "x" N B) (raw u "dy" N B) (raw u "w" 0 B)
        ((u.mem "mu" (u.pids 0)).read .real) ((u.mem "r" (u.pids 0)).read .real) i.1)
      (fun i => i.1.val<N) u (j,PUnit.unit) hj (row_injective _ _)
  · intro q o outside
    rw [hm]
    apply commit_frame
    by_cases hq : q="dx"
    · exact Or.inr (fun i _ hi ho => outside ⟨hq,i.1,hi,ho⟩)
    · exact Or.inl hq

theorem affine_complete (M : Algebra α) (rows N wc bc : Nat) (bias : Bool) (u : State α) :
    ∃ v, FP.Structural.exec (engine M) (KernelAffine.kernel rows N wc bc bias) u=some v ∧
      v.mem=KernelAffine.heap M rows N wc bc (u.pids 0) bias u ∧
      (∀ j : Fin 128,u.pids 0*128+j.val<wc → v.mem "dw" (u.pids 0*128+j.val)=
        Cell.mk .real (KernelReference.cast M (KernelAffine.carry M N (u.pids 0) u rows (j,PUnit.unit)).1)) ∧
      (bias=Bool.true → ∀ j : Fin 128,u.pids 0*128+j.val<bc → v.mem "db" (u.pids 0*128+j.val)=
        Cell.mk .real (KernelReference.cast M (KernelAffine.carry M N (u.pids 0) u rows (j,PUnit.unit)).2)) ∧
      (∀ q o,¬affineFootprint wc bc (u.pids 0) bias q o → v.mem q o=u.mem q o) := by
  obtain ⟨v,he,hm⟩ := KernelAffine.source_memory M rows N wc bc bias u
  let midmem := commit "dw" (fun i : TileIndex [128] => u.pids 0*128+i.1.val)
      (fun i => Cell.mk .real (KernelReference.cast M (KernelAffine.carry M N (u.pids 0) u rows i).1))
      (fun i => u.pids 0*128+i.1.val<wc) (TileShape.allIndices [128]) u.mem
  let middle : State α := {u with mem := midmem}
  refine ⟨v,he,hm,?_,?_,?_⟩
  · intro j hj
    rw [hm]
    cases bias
    · exact commit_readback "dw" (fun i : TileIndex [128] => u.pids 0*128+i.1.val)
        (fun i => Cell.mk .real (KernelReference.cast M (KernelAffine.carry M N (u.pids 0) u rows i).1))
        (fun i => u.pids 0*128+i.1.val<wc) u (j,PUnit.unit) hj (row_injective _ _)
    · change commit "db" _ _ _ _ middle.mem _ _=_
      rw [commit_frame _ _ _ _ _ middle _ _ (Or.inl (by decide))]
      exact commit_readback "dw" (fun i : TileIndex [128] => u.pids 0*128+i.1.val)
        (fun i => Cell.mk .real (KernelReference.cast M (KernelAffine.carry M N (u.pids 0) u rows i).1))
        (fun i => u.pids 0*128+i.1.val<wc) u (j,PUnit.unit) hj (row_injective _ _)
  · intro hb j hj
    subst bias
    rw [hm]
    exact commit_readback "db" (fun i : TileIndex [128] => u.pids 0*128+i.1.val)
      (fun i => Cell.mk .real (KernelReference.cast M (KernelAffine.carry M N (u.pids 0) u rows i).2))
      (fun i => u.pids 0*128+i.1.val<bc) middle (j,PUnit.unit) hj (row_injective _ _)
  · intro q o outside
    have fw : middle.mem q o=u.mem q o := by
      apply commit_frame
      by_cases hq : q="dw"
      · exact Or.inr (fun i _ hi ho => outside (Or.inl ⟨hq,i.1,hi,ho⟩))
      · exact Or.inl hq
    rw [hm]
    cases bias
    · exact fw
    · change commit "db" _ _ _ _ middle.mem q o=_
      rw [commit_frame]
      · exact fw
      · by_cases hq : q="db"
        · exact Or.inr (fun i _ hi ho => outside (Or.inr ⟨rfl,hq,i.1,hi,ho⟩))
        · exact Or.inl hq

#axiomsClean forward_complete
#axiomsClean backward_complete
#axiomsClean affine_complete
end
end VeriTile.Bench.Repairs.FastLayernorm.KernelOutputs
