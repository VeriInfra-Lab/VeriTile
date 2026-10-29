import bench.tritonbench_g.triton_argmax.Proofs.DimBackend
namespace VeriTile.Bench.TritonBenchG.TritonArgmax.DimCorrect
open VeriTile.Triton FP.Structural TypedSource MatrixReduction.MachineMax
open MatrixReduction.ConcreteCorrect (Word channel)
open DimBackend
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

def row (s : State Word) (BM : Nat) (i : Fin BM) : Nat := s.pids 0 * BM + i.val

def inputOffset (s : State Word) (N K BM BN : Nat) (i : Fin BM) (j : Fin BN) : Nat :=
  row s BM i * N * K + j.val * K + s.pids 1

def outputOffset (s : State Word) (K BM : Nat) (i : Fin BM) : Nat := row s BM i * K + s.pids 1

def padding : Storage → Word
  | .fp16 => (MatrixReduction.BinaryConversion.narrow (negativeInfinity .fp32)).zeroExtend 32
  | .fp32 => negativeInfinity .fp32

def loaded (storage : Storage) (s : State Word) (X : RegionName)
    (M N K BM BN : Nat) (i : Fin BM) (j : Fin BN) : Word :=
  if row s BM i < M ∧ j.val < N then
    toFloat (channel storage) ((s.mem X (inputOffset s N K BM BN i j)).read (channel storage).toTileDType)
  else padding storage

def input (storage : Storage) (s : State Word) (X : RegionName)
    (M N K BM BN : Nat) (i : Fin BM) (j : Fin BN) : Word :=
  DimBackend.normalize (channel storage) (loaded storage s X M N K BM BN i j)

def selected (storage : Storage) (s : State Word) (X : RegionName)
    (M N K BM BN : Nat) (hB : 0 < BN) (i : Fin BM) : Fin BN :=
  best .fp32 (input storage s X M N K BM BN i) hB

def output (storage : Storage) (s : State Word) (X : RegionName)
    (M N K BM BN : Nat) (hB : 0 < BN) (i : Fin BM) : Int :=
  let j := selected storage s X M N K BM BN hB i
  if rank .fp32 (negativeInfinity .fp32) < rank .fp32 (input storage s X M N K BM BN i j)
    then Int.ofNat j.val else 0

def written (storage : Storage) (s : State Word) (X : RegionName) (O : Region .int)
    (M N K BM BN : Nat) (hB : 0 < BN) : State Word :=
  (TileShape.allIndices [BM]).foldl (fun t i =>
    if row s BM i.1 < M then
      t.write (Region.cast O) (outputOffset s K BM i.1)
        (.mk .int (output storage s X M N K BM BN hB i.1)) else t) s

private theorem fold_memory {α ι : Type} (l : List ι) (r : RegionName)
    (off : ι → Nat) (value : ι → Cell α) (active : ι → Prop) [DecidablePred active] (s : State α) :
    (l.foldl (fun t i => if active i then t.write r (off i) (value i) else t) s).mem =
      l.foldl (fun mem i => fun r' o =>
        if active i ∧ r' = r ∧ o = off i then value i else mem r' o) s.mem := by
  induction l generalizing s with
  | nil => rfl
  | cons i l ih =>
    by_cases hi : active i
    · simp only [List.foldl_cons, if_pos hi, ih]
      congr 1
      funext r' o
      simp [hi, State.write]
    · simp only [List.foldl_cons, if_neg hi, ih]
      congr 1
      funext r' o
      simp [hi]

theorem source_run (storage : Storage) (rest : Algebra Word) (s : State Word)
    (X : RegionName) (O : Region .int) (M N K BM BN : Nat) (index64 : Bool)
    (hN : 0 < N) (hBN : N ≤ BN) :
    ∃ t, FP.Structural.exec (backend rest) (dim storage X O M N K BM BN index64) s = some t ∧
      t.mem = (written storage s X O M N K BM BN (hN.trans_le hBN)).mem := by
  have hB : 0 < BN := hN.trans_le hBN
  have hBne : BN ≠ 0 := Nat.ne_of_gt hB
  have hDone : ¬ BN < N := Nat.not_lt.mpr hBN
  cases storage <;> cases index64 <;>
    simp [dim, ComputeKernel.surfaceBody, FP.Structural.exec, run, step,
      range_step, range_done, hN, hBN, hBne, hDone,
      evalExpr, evalComputeOp, evalOp_unfold, store, backend,
      MatrixReduction.ConcreteCorrect.backend, indexed, DimBackend.normalize,
      State.setReg, TileShape.axisDim, hB, TileShape.insertAxisIndex,
      TileShape.eraseAxis, selected, input, loaded, output, padding, written,
      row, inputOffset, outputOffset, channel,
      ofFloat, toFloat, Region.cast, ComputeDType.eraseDType, bop, numeric,
      numericLt, natLt, Nat.ne_of_gt hB, TileShape.allIndices, Option.bind]
  all_goals simp only [fold_memory]
  all_goals rfl

@[simp] theorem normalized_padding (storage : Storage) :
    DimBackend.normalize (channel storage) (padding storage) = negativeInfinity .fp32 := by
  cases storage
  · decide
  · rfl

theorem input_padding (storage : Storage) (s : State Word) (X : RegionName)
    (M N K BM BN : Nat) (i : Fin BM) (j : Fin BN) (hj : N ≤ j.val) :
    input storage s X M N K BM BN i j = negativeInfinity .fp32 := by
  simp only [input, loaded, if_neg (show ¬ (row s BM i < M ∧ j.val < N) by omega), normalized_padding]

theorem output_eq_selected (storage : Storage) (s : State Word) (X : RegionName)
    (M N K BM BN : Nat) (hB : 0 < BN) (i : Fin BM)
    (hnotNaN : ∀ j, isNaN .fp32 (input storage s X M N K BM BN i j) = Bool.false) :
    output storage s X M N K BM BN hB i =
      Int.ofNat (selected storage s X M N K BM BN hB i).val := by
  dsimp only [output]
  split
  · rfl
  · rename_i h
    have hl := negativeInfinity_le (input storage s X M N K BM BN i ⟨0, hB⟩) (hnotNaN _)
    have hm := best_maximal .fp32 (input storage s X M N K BM BN i) hB ⟨0, hB⟩
    have he : rank .fp32 (input storage s X M N K BM BN i ⟨0, hB⟩) =
        rank .fp32 (input storage s X M N K BM BN i (selected storage s X M N K BM BN hB i)) := by
      change _ ≤ rank .fp32 (input storage s X M N K BM BN i (selected storage s X M N K BM BN hB i)) at hm
      omega
    have hi := best_leftmost .fp32 (input storage s X M N K BM BN i) hB ⟨0, hB⟩ he
    have hj : (selected storage s X M N K BM BN hB i).val = 0 := by
      change (selected storage s X M N K BM BN hB i).val ≤ 0 at hi
      omega
    rw [hj]
    rfl

theorem selected_valid (storage : Storage) (s : State Word) (X : RegionName)
    (M N K BM BN : Nat) (hN : 0 < N) (hBN : N ≤ BN) (i : Fin BM)
    (hnotNaN : ∀ j, isNaN .fp32 (input storage s X M N K BM BN i j) = Bool.false) :
    (selected storage s X M N K BM BN (hN.trans_le hBN) i).val < N := by
  let j := selected storage s X M N K BM BN (hN.trans_le hBN) i
  by_contra hn
  have hp := input_padding storage s X M N K BM BN i j (by omega)
  have hl := negativeInfinity_le (input storage s X M N K BM BN i ⟨0, hN.trans_le hBN⟩) (hnotNaN _)
  have hm := best_maximal .fp32 (input storage s X M N K BM BN i) (hN.trans_le hBN) ⟨0, hN.trans_le hBN⟩
  change _ ≤ rank .fp32 (input storage s X M N K BM BN i j) at hm
  have he : rank .fp32 (input storage s X M N K BM BN i ⟨0, hN.trans_le hBN⟩) =
      rank .fp32 (input storage s X M N K BM BN i j) := by rw [hp] at hm ⊢; omega
  have hi := best_leftmost .fp32 (input storage s X M N K BM BN i) (hN.trans_le hBN) ⟨0, hN.trans_le hBN⟩ he
  change j.val ≤ 0 at hi
  change ¬ j.val < N at hn
  omega

/-- Original dimension-specific source in its accepted one-block regime.
Every active row returns a valid original-axis index, maximal by machine value
and least index on ties. The stored Int is an index payload; native machine
integer/address overflow and compiler refinement are separate obligations. -/
theorem source_correct (storage : Storage) (rest : Algebra Word) (s : State Word)
    (X : RegionName) (O : Region .int) (M N K BM BN : Nat) (index64 : Bool)
    (hN : 0 < N) (hBN : N ≤ BN)
    (hOutInj : Function.Injective (outputOffset s K BM))
    (hnotNaN : ∀ i j, isNaN .fp32 (input storage s X M N K BM BN i j) = Bool.false) :
    ∃ t, FP.Structural.exec (backend rest) (dim storage X O M N K BM BN index64) s = some t ∧
      (∀ i : Fin BM, row s BM i < M → ∃ j : Fin BN,
        j.val < N ∧ BitCorrect.IsArgMaximum .fp32 (input storage s X M N K BM BN i) j ∧
        t.mem (Region.cast O) (outputOffset s K BM i) = .mk .int (Int.ofNat j.val)) ∧
      (∀ r o, (r ≠ Region.cast O ∨ ∀ i : Fin BM, row s BM i < M → o ≠ outputOffset s K BM i) →
        t.mem r o = s.mem r o) := by
  obtain ⟨t, he, hm⟩ := source_run storage rest s X O M N K BM BN index64 hN hBN
  refine ⟨t, he, ?_, ?_⟩
  · intro i hi
    refine ⟨selected storage s X M N K BM BN (hN.trans_le hBN) i,
      selected_valid storage s X M N K BM BN hN hBN i (hnotNaN i),
      ⟨best_maximal .fp32 _ _, best_leftmost .fp32 _ _⟩, ?_⟩
    rw [hm]
    unfold written
    rw [State.masked_scatter_readback (Region.cast O)
      (fun k : TileIndex [BM] => outputOffset s K BM k.1) _
      (fun k : TileIndex [BM] => row s BM k.1 < M) s (i, PUnit.unit) hi]
    · rw [output_eq_selected storage s X M N K BM BN _ i (hnotNaN i)]
    · rintro ⟨a, u⟩ ⟨b, v⟩ hab
      have hh : a = b := hOutInj hab
      subst b
      cases u; cases v
      rfl
  · intro r o hmiss
    rw [hm]
    unfold written
    apply State.masked_scatter_frame
    rcases hmiss with hr | ho
    · exact Or.inl hr
    · exact Or.inr (fun i _ hi => ho i.1 hi)

#axiomsClean output_eq_selected
#axiomsClean selected_valid
#axiomsClean source_correct
#axiomsClean source_run
end
end VeriTile.Bench.TritonBenchG.TritonArgmax.DimCorrect
