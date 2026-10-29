import bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.KernelStorage
namespace VeriTile.Bench.Repairs.Conv2d.KernelStores
open VeriTile.Triton FP.Structural KernelBinding KernelGrid
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 16384
set_option linter.unusedSimpArgs false

variable {α : Type} [Inhabited α]

def stores (M : Algebra α) (H : KernelHalf.Dot α) (D : KernelDefault.Dot α)
    (s : Shape) (f : Flags) (pid : Program) (t : State α) : State α :=
  (TileShape.allIndices [128,32]).foldl
    (fun u i=>if KernelHalf.active (parameters s f) (atProgram pid t) i then
      u.write "output" (KernelHalf.outputOffset (parameters s f) (atProgram pid t) i)
        (.mk .fp32 (KernelGrid.value M H D s f pid t i)) else u) (atProgram pid t)

/-- All program memory effects are the actual shared DSL's masked store fold,
including the exact source lane order. -/
theorem program_stores (M : Algebra α) (H : KernelHalf.Dot α) (D : KernelDefault.Dot α)
    (s : Shape) (v : Valid s) (f : Flags) (pid : Program) (hg : pid.2.2<s.groups) (t : State α) :
    ∃ u,KernelGrid.program M H D s f pid t=some u ∧ u.mem=(stores M H D s f pid t).mem := by
  have hw:=stores_unique s v f pid hg t
  cases hf:f.fp16
  · obtain ⟨u,he,hm,hp⟩:=KernelDefault.correctness M D (parameters s f) (atProgram pid t) hw
    refine ⟨u,by simpa [KernelGrid.program,KernelGrid.backend,KernelSource.kernel,hf] using he,?_⟩
    simpa [stores,KernelDefault.written,KernelGrid.value,hf] using hm
  · obtain ⟨u,he,hm,hp⟩:=KernelHalf.correctness M H (parameters s f) (atProgram pid t) hw
    refine ⟨u,by simpa [KernelGrid.program,KernelGrid.backend,KernelSource.kernel,hf] using he,?_⟩
    simpa [stores,KernelHalf.written,KernelGrid.value,hf] using hm

/-- Per-program byteheap semantics, derived from source execution and local
store commutation. This holds for every preceding heap/state and every
program prefix; no final-heap observation gate appears. -/
theorem program_bytes (instructions : KernelNative.Instructions) (s : Shape) (v : Valid s)
    (f : Flags) (pid : Program) (hg : pid.2.2<s.groups) (base : Nat) (before : KernelStorage.Heap)
    (t : State KernelStorage.Value) :
    let o:=KernelNative.operations instructions f
    ∃ u,KernelGrid.program (KernelNumeric.algebra o) (KernelNumeric.halfDot o)
      (KernelNumeric.defaultDot o) s f pid t=some u ∧
      KernelStorage.image o base s.size before u.mem=
        (TileShape.allIndices [128,32]).foldl
          (fun heap i=>if KernelHalf.active (parameters s f) (atProgram pid t) i then
            Physical.write heap (base+4*KernelHalf.outputOffset (parameters s f) (atProgram pid t) i)
              (KernelNumeric.scalar o (KernelGrid.value (KernelNumeric.algebra o) (KernelNumeric.halfDot o)
                (KernelNumeric.defaultDot o) s f pid t i)) else heap)
          (KernelStorage.image o base s.size before t.mem) := by
  dsimp only
  let o:=KernelNative.operations instructions f
  obtain ⟨u,he,hm⟩:=program_stores (KernelNumeric.algebra o) (KernelNumeric.halfDot o)
    (KernelNumeric.defaultDot o) s v f pid hg t
  refine ⟨u,he,?_⟩
  rw [hm]
  unfold stores
  apply KernelStorage.fold_projection
  intro i hi ha
  rw [output_address s v f]
  exact address_bound s v _ ((active_iff s v f pid hg t i).mp ha)

#axiomsClean program_stores
#axiomsClean program_bytes
end
end VeriTile.Bench.Repairs.Conv2d.KernelStores
