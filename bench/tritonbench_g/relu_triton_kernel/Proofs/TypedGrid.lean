import bench.tritonbench_g.relu_triton_kernel.Proofs.TypedValue

namespace VeriTile.Bench.TritonBenchG.ReluTritonKernel.TypedValue
open VeriTile.Triton FP.Structural GatedGrid
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

def reference {α : Type} [Inhabited α] (M : Algebra α)
    (cmp16 cmp32 : α → α → Bool) (k : Kind) (x y : RegionName)
    (N B : Nat) (mem : Memory α) : Memory α :=
  fun r o => if r = y ∧ o < B ∧ o < N then
    .mk .fp32 (resultValue M cmp16 cmp32 k (mem x o)) else mem r o

theorem pid_memory {α : Type} [Inhabited α] (M : Algebra α)
    (cmp16 cmp32 : α → α → Bool) (k : Kind) (x y : RegionName)
    (N B : Nat) (s : State α) (pid : Nat) :
    ∃ t, FP.Structural.exec (backend M cmp16 cmp32) (materialized k x y N B) (atPid s pid) = some t ∧
      t.mem = if pid = 0 then reference M cmp16 cmp32 k x y N B s.mem else s.mem := by
  obtain ⟨t, ht, hm⟩ := source_run M cmp16 cmp32 k x y N B (atPid s pid)
  refine ⟨t, ht, ?_⟩
  rw [hm]
  by_cases hp : pid = 0
  · subst pid
    simp only [atPid, outputState, ↓reduceIte, Nat.zero_mul, Nat.zero_add]
    funext r o
    by_cases h : r = y ∧ o < B ∧ o < N
    · have ho : o < B := h.2.1
      let i : TileIndex [B] := (⟨o, ho⟩, PUnit.unit)
      have hr : r = y := h.1
      subst r
      rw [show o = i.1.val from rfl]
      rw [State.masked_scatter_readback]
      · simp [reference, i, h.2.1, h.2.2]
      · exact h.2.2
      · intro a b hab
        rcases a with ⟨a, ⟨⟩⟩
        rcases b with ⟨b, ⟨⟩⟩
        have : a = b := Fin.ext hab
        subst b
        rfl
    · rw [show reference M cmp16 cmp32 k x y N B s.mem r o = s.mem r o by
        simp only [reference, h, ↓reduceIte]]
      apply State.masked_scatter_frame
      by_cases hr : r = y
      · right
        intro i _ hi heq
        exact h ⟨hr, heq ▸ i.1.isLt, heq ▸ hi⟩
      · exact Or.inl hr
  · simp [atPid, outputState, hp]

theorem prefix_correct {α : Type} [Inhabited α] (M : Algebra α)
    (cmp16 cmp32 : α → α → Bool) (p : Profile) (x y : RegionName)
    (N B : Nat) (initial : State α) (count : Nat) :
    ∃ final, runPrefix (backend M cmp16 cmp32) (source p x y N B) initial count = some final ∧
      final.mem = if count = 0 then initial.mem
        else reference M cmp16 cmp32 p.kind x y N B initial.mem :=
  prefix_of_first_only (backend M cmp16 cmp32) _ _
    (pid_memory M cmp16 cmp32 p.kind x y N B) initial count

theorem grid_correct {α : Type} [Inhabited α] (M : Algebra α)
    (cmp16 cmp32 : α → α → Bool) (p : Profile) (x y : RegionName)
    (N B : Nat) (initial : State α) :
    ∃ final, runGrid (backend M cmp16 cmp32) (source p x y N B) N B initial = some final ∧
      final.mem = if gridCount N B = 0 then initial.mem
        else reference M cmp16 cmp32 p.kind x y N B initial.mem :=
  prefix_correct M cmp16 cmp32 p x y N B initial (gridCount N B)

theorem reference_frame {α : Type} [Inhabited α] (M : Algebra α)
    (cmp16 cmp32 : α → α → Bool) (k : Kind) (x y : RegionName)
    (N B : Nat) (mem : Memory α) (r : RegionName) (o : Nat)
    (outside : r ≠ y ∨ B ≤ o ∨ N ≤ o) :
    reference M cmp16 cmp32 k x y N B mem r o = mem r o := by
  simp only [reference]
  rw [if_neg]
  intro h
  rcases outside with hr | hb | hn
  · exact hr h.1
  · omega
  · omega

/-- The three floating materializations preserve the erased original body.
Integer/bool lowering uses a separate explicit conversion route and is not
misrepresented by this floating erasure theorem. -/
theorem floating_source_projection (k : Kind)
    (hk : k = .fp16 ∨ k = .bf16 ∨ k = .fp32)
    (x y : RegionName) (N B : Nat) :
    (materialized k x y N B).eraseDType.toAlgKernel.body =
      (relu_kernel x y N B).toAlgKernel.body := by
  rcases hk with rfl | rfl | rfl
  all_goals simp only [materialized, Kind.inputType, Kind.resultType,
    resultOp, storedOp, relu_kernel, ComputeKernel.eraseDType, ComputeKernel.surfaceBody,
    ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
    FloatDType.toTileDType, FloatDType.numericDType, NumericDType.eraseDType,
    ComparableDType.eraseDType, VeriTile.Triton.eraseDType, ComputeDType.eraseDType,
    Region.cast, Except.bind, bind, List.map_cons, List.map_nil]
  all_goals norm_num

#specNonCircular reference avoiding [source, materialized, runPrefix, runGrid, FP.Structural.exec]
#axiomsClean pid_memory
#axiomsClean prefix_correct
#axiomsClean grid_correct
#axiomsClean reference_frame
#axiomsClean floating_source_projection
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ReluTritonKernel.TypedValue
