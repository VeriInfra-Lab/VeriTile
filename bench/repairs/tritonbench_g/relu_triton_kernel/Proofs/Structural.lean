import bench.repairs.tritonbench_g.relu_triton_kernel.Kernels
import bench.tritonbench_g.relu_triton_kernel.Proofs.FP
import VeriTile.Meta.StatementAudit

/-! Reuses the original ReLU scatter lemma. The tile-to-grid induction is the
contiguous specialization of the prior repair's StructuralGrid proof; it allows
any permutation of the launch and proves results from entry-state input cells. -/
namespace VeriTile.Bench.Repairs.ReluTritonKernel.Structural
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

def scalar {α : Type} (M : Algebra α) (cmp : α → α → Bool) (x : α) : α :=
  if cmp (M.literal (some .fp32) .real 0) x then x else M.literal (some .fp32) .real 0

def outputState {α : Type} [Inhabited α] (M : Algebra α) (cmp : α → α → Bool)
    (input output : RegionName) (ib ob W N B : Nat) (s : State α) : State α :=
  (TileShape.allIndices [B]).foldl (fun t i =>
    if s.pids 0*B+i.1.val<N then
      t.write output (ob+W*(s.pids 0*B+i.1.val))
        (.mk .real (scalar M cmp (M.fp32Load ((s.mem input (ib+W*(s.pids 0*B+i.1.val))).read .real))))
    else t) s

theorem source_run {α : Type} [Inhabited α] (M : Algebra α)
    (cmp : α → α → Bool) (hc : M.compareLe (some .fp32) .real=some cmp)
    (input output : RegionName) (ib ob W N B : Nat) (s : State α) :
    ∃ t, FP.Structural.exec M (kernel input output ib ob W N B) s=some t ∧
      t.mem=(outputState M cmp input output ib ob W N B s).mem := by
  simp [kernel, FP.Structural.exec, run, step, evalExpr, evalComputeOp, evalOp_unfold,
    Option.bind, ComputeDType.eraseDType, numeric, numericLe, numericLt, natLt, FP.Structural.bop, store, hc,
    outputState, scalar, Region.cast, ofFloat, toFloat]
  simp +contextual only [ite_true]
  norm_num only
  apply VeriTile.Bench.TritonBenchG.ReluTritonKernel.FPProofs.scatter_mem_congr
  rfl

theorem fused_run {α : Type} [Inhabited α] (M : Algebra α)
    (cmp : α → α → Bool) (hc : M.compareLe (some .fp32) .real=some cmp)
    (input output : RegionName) (ib ob W N B : Nat) (s : State α) :
    ∃ t, FP.Structural.exec M (fusedKernel input output ib ob W N B) s=some t ∧
      t.mem=(outputState M cmp input output ib ob W N B s).mem := by
  simp [fusedKernel, FP.Structural.exec, run, step, evalExpr, evalComputeOp, evalOp_unfold,
    Option.bind, ComputeDType.eraseDType, numeric, numericLe, numericLt, natLt, FP.Structural.bop, store, hc,
    outputState, scalar, Region.cast, ofFloat, toFloat]
  simp +contextual only [ite_true]
  norm_num only
  apply VeriTile.Bench.TritonBenchG.ReluTritonKernel.FPProofs.scatter_mem_congr
  rfl

/-- Every cell outside the actually active store lanes is preserved. -/
theorem outputState_frame {α : Type} [Inhabited α] (M : Algebra α)
    (cmp : α → α → Bool) (input output : RegionName) (ib ob W N B : Nat)
    (s : State α) (r : RegionName) (address : Nat)
    (outside : r ≠ output ∨ ∀ i : Fin B, s.pids 0 * B + i.val < N →
      address ≠ ob + W * (s.pids 0 * B + i.val)) :
    (outputState M cmp input output ib ob W N B s).mem r address = s.mem r address := by
  unfold outputState
  apply State.masked_scatter_frame
  rcases outside with hr | ha
  · exact Or.inl hr
  · exact Or.inr (fun i _ hi => ha i.1 hi)

def atPid {α : Type} (s : State α) (pid : Nat) : State α :=
  {s with pids:=fun ax => if ax=0 then pid else s.pids ax}

def tileExec {α : Type} [Inhabited α] (M : Algebra α)
    (input output : RegionName) (ib ob W N B : Nat) (pid : Nat) (s : State α) : Option (State α) :=
  FP.Structural.exec M (kernel input output ib ob W N B) (atPid s pid)

def TileOutside (output : RegionName) (ob W N pid B : Nat) (r : RegionName) (a : Nat) : Prop :=
  r ≠ output ∨ ∀ lane : TileIndex [B], pid*B+lane.1.val<N → a≠ob+W*(pid*B+lane.1.val)

theorem tile_contract {α : Type} [Inhabited α] (M : Algebra α)
    (cmp : α → α → Bool) (hc : M.compareLe (some .fp32) .real=some cmp)
    (input output : RegionName) (ib ob W N B : Nat) (hW : 0<W) (pid : Nat) (s : State α) :
    ∃ t, tileExec M input output ib ob W N B pid s=some t ∧
      (∀ lane : TileIndex [B], pid*B+lane.1.val<N →
        t.mem output (ob+W*(pid*B+lane.1.val))=
          .mk .real (scalar M cmp (M.fp32Load ((s.mem input (ib+W*(pid*B+lane.1.val))).read .real)))) ∧
      (∀ r a, TileOutside output ob W N pid B r a → t.mem r a=s.mem r a) := by
  obtain ⟨t,hr,hm⟩:=source_run M cmp hc input output ib ob W N B (atPid s pid)
  refine ⟨t,hr,?_,?_⟩
  · intro i hi
    rw [hm]
    simp only [outputState,atPid,↓reduceIte]
    apply State.masked_scatter_readback output
      (fun j : TileIndex [B]=>ob+W*(pid*B+j.1.val))
      (fun j=>Cell.mk .real (scalar M cmp (M.fp32Load ((s.mem input (ib+W*(pid*B+j.1.val))).read .real))))
      (fun j=>pid*B+j.1.val<N) (atPid s pid) i hi
    intro a b hab
    dsimp at hab
    have h := Nat.eq_of_mul_eq_mul_left hW (Nat.add_left_cancel hab)
    apply Prod.ext
    · apply Fin.ext
      omega
    · exact Subsingleton.elim _ _
  · intro r a outside
    rw [hm]
    simp only [outputState,atPid,↓reduceIte]
    apply State.masked_scatter_frame
    rcases outside with hr | ha
    · exact Or.inl hr
    · exact Or.inr (fun lane _ hl => ha lane hl)

def Covered (B : Nat) (tiles : List Nat) (i : Nat) : Prop :=
  ∃ t ∈ tiles, ∃ lane : TileIndex [B], i=t*B+lane.1.val

def runTiles {α : Type} (tile : Nat → State α → Option (State α)) :
    List Nat → State α → Option (State α)
  | [],s => some s
  | t::tiles,s => (tile t s).bind (runTiles tile tiles)

def Separate (input output : RegionName) (ib ob W n : Nat) : Prop :=
  ∀ i,i<n → ∀ j,j<n → input≠output ∨ ib+W*i≠ob+W*j

theorem tiles_success {α : Type} [Inhabited α] (M : Algebra α)
    (cmp : α → α → Bool) (hc : M.compareLe (some .fp32) .real=some cmp)
    (input output : RegionName) (ib ob W n B : Nat) (hW : 0<W)
    (sep : Separate input output ib ob W n) (tiles : List Nat) (s : State α) :
    ∃ final, runTiles (tileExec M input output ib ob W n B) tiles s=some final ∧
      (∀ i,i<n → Covered B tiles i → final.mem output (ob+W*i)=
        .mk .real (scalar M cmp (M.fp32Load ((s.mem input (ib+W*i)).read .real)))) ∧
      (∀ i,i<n → final.mem input (ib+W*i)=s.mem input (ib+W*i)) ∧
      (∀ r a,(∀ t∈tiles, TileOutside output ob W n t B r a) → final.mem r a=s.mem r a) := by
  induction tiles generalizing s with
  | nil =>
    refine ⟨s,rfl,?_,by intros; rfl,by intros; rfl⟩
    rintro i hi ⟨t,ht,_⟩
    simp at ht
  | cons t tiles ih =>
    obtain ⟨middle,hexec,hvalue,hframe⟩ := tile_contract M cmp hc input output ib ob W n B hW t s
    obtain ⟨final,htail,htailValue,htailInput,htailFrame⟩ := ih middle
    have hinput (i : Nat) (hi : i<n) : middle.mem input (ib+W*i)=s.mem input (ib+W*i) := by
      apply hframe
      by_cases hr : input=output
      · right
        intro lane hl
        exact (sep i hi _ hl).resolve_left (by simpa [hr])
      · exact Or.inl hr
    refine ⟨final,by simp only [runTiles,hexec,Option.bind_some,htail],?_,?_,?_⟩
    · intro i hi covered
      by_cases tail : Covered B tiles i
      · rw [htailValue i hi tail,hinput i hi]
      · have keep : final.mem output (ob+W*i)=middle.mem output (ob+W*i) := by
          apply htailFrame
          intro other ho
          right
          intro lane hl heq
          have eqi := Nat.eq_of_mul_eq_mul_left hW (Nat.add_left_cancel heq)
          exact tail ⟨other,ho,lane,eqi⟩
        rw [keep]
        obtain ⟨other,ho,lane,heq⟩:=covered
        rcases List.mem_cons.mp ho with eq | hm
        · subst other
          subst i
          exact hvalue lane hi
        · exact False.elim (tail ⟨other,hm,lane,heq⟩)
    · intro i hi
      rw [htailInput i hi,hinput i hi]
    · intro r a outside
      rw [htailFrame r a (fun other hm => outside other (List.mem_cons_of_mem t hm))]
      exact hframe r a (outside t List.mem_cons_self)

structure Schedule (n : Nat) where
  programs : List Nat
  complete : programs.Perm (List.range (gridCount n))

def orderedSchedule (n : Nat) : Schedule n := ⟨List.range (gridCount n),List.Perm.refl _⟩

theorem covered (n i : Nat) (hi : i<n) (schedule : Schedule n) :
    Covered 1024 schedule.programs i := by
  refine ⟨i/1024,schedule.complete.mem_iff.mpr (List.mem_range.mpr ?_),
    ⟨⟨i%1024,Nat.mod_lt _ (by decide)⟩,PUnit.unit⟩,?_⟩
  · unfold gridCount
    omega
  · simpa only [Nat.mul_comm] using (Nat.div_add_mod i 1024).symm

def runGrid {α : Type} [Inhabited α] (M : Algebra α)
    (input output : RegionName) (ib ob W n : Nat) (schedule : Schedule n) (s : State α) :=
  runTiles (tileExec M input output ib ob W n 1024) schedule.programs s

theorem grid_complete {α : Type} [Inhabited α] (M : Algebra α)
    (cmp : α → α → Bool) (hc : M.compareLe (some .fp32) .real=some cmp)
    (input output : RegionName) (ib ob W n : Nat) (hW : 0<W)
    (sep : Separate input output ib ob W n) (schedule : Schedule n) (s : State α) :
    ∃ t, runGrid M input output ib ob W n schedule s=some t ∧
      (∀ i,i<n → t.mem output (ob+W*i)=.mk .real (scalar M cmp (M.fp32Load ((s.mem input (ib+W*i)).read .real)))) ∧
      (∀ r a,(r≠output ∨ ∀ i,i<n → a≠ob+W*i) → t.mem r a=s.mem r a) := by
  obtain ⟨t,he,hv,_,hf⟩ := tiles_success M cmp hc input output ib ob W n 1024 hW sep schedule.programs s
  refine ⟨t,he,fun i hi=>hv i hi (covered n i hi schedule),?_⟩
  intro r a outside
  apply hf
  intro pid hp
  rcases outside with hr | ha
  · exact Or.inl hr
  · exact Or.inr (fun lane hl=>ha _ hl)

#axiomsClean source_run
#axiomsClean grid_complete
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ReluTritonKernel.Structural
