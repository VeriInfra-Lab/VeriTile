import bench.repairs.tritonbench_g.int8_matmul_kernel_modular_checked.Proofs.GridExecution

namespace VeriTile.Bench.Repairs.Int8MatmulKernelModularChecked.TileModel
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.Int8MatmulKernel
open VeriTile.Bench.TritonBenchG.Int8MatmulKernel.Whole001
noncomputable section
set_option maxHeartbeats 8000000

theorem fixed_cell {P A V : Type} (writes : P → A → Prop) [∀ p, DecidablePred (writes p)]
    (value : A → V) (schedule : List P) (memory : A → V) (loc : A) :
    Schedule.run (Schedule.put writes (fun _ => value)) schedule memory loc =
      if ∃ pid ∈ schedule, writes pid loc then value loc else memory loc := by
  classical
  induction schedule generalizing memory with
  | nil => simp [Schedule.run]
  | cons pid tail ih =>
    change Schedule.run (Schedule.put writes (fun _ => value)) tail
      (Schedule.put writes (fun _ => value) pid memory) loc = _
    rw [ih]
    by_cases ht : ∃ p ∈ tail, writes p loc
    · simp [ht]
    · by_cases hp : writes pid loc <;> simp [ht,hp,Schedule.put]

theorem source_row (layout : Layout) (s : Shape) (c : Fin 15) (initial : State Word32)
    (pid : Nat) (i : CIndex (parameters layout s c)) :
    outputRow (parameters layout s c) (tileState initial pid) i = row s c pid i.1.val := by rfl
theorem source_column (layout : Layout) (s : Shape) (c : Fin 15) (initial : State Word32)
    (pid : Nat) (i : CIndex (parameters layout s c)) :
    outputColumn (parameters layout s c) (tileState initial pid) i = col s c pid i.2.1.val := by rfl

theorem grid_covered (layout : Layout) (s : Shape) (c : Fin 15) (initial : State Word32)
    (hout : OutputLayout layout s) (m n : Nat) (hm : m < s.m) (hn : n < s.n) :
    ∃ pid ∈ List.range (s.programs c),
      tileWrites layout s c initial pid (layout.output,Whole001.outputOffset s m n) := by
  obtain ⟨pid,hp,r,col,hr,hcol,hm',hn'⟩ := source_cell_covered s c m n hm hn
  let i : CIndex (parameters layout s c) := (⟨r,hr⟩,⟨col,hcol⟩,PUnit.unit)
  refine ⟨pid,by simpa using hp,i,?_,?_⟩
  · change row s c pid r < s.m ∧ Whole001.col s c pid col < s.n
    simpa [hm',hn'] using And.intro hm hn
  · apply congrArg (fun off => (layout.output,off))
    rw [output_coordinates_source layout s c (tileState initial pid) i hout,
      source_row,source_column]
    simp [i,hm',hn']

theorem writes_inside (layout : Layout) (s : Shape) (c : Fin 15) (initial : State Word32)
    (hout : OutputLayout layout s) (pid : Nat) (loc : RegionName × Nat)
    (write : tileWrites layout s c initial pid loc) :
    loc.1 = layout.output ∧ loc.2 < s.m*s.n := by
  obtain ⟨i,hi,rfl⟩ := write
  refine ⟨rfl,?_⟩
  rw [output_coordinates_source layout s c (tileState initial pid) i hout]
  have npos : 0 < s.n := Nat.lt_of_le_of_lt (Nat.zero_le _) hi.2
  change outputRow (parameters layout s c) (tileState initial pid) i < s.m ∧
    outputColumn (parameters layout s c) (tileState initial pid) i < s.n at hi
  unfold Whole001.outputOffset
  rw [max_eq_left npos]
  have bound := Nat.mul_le_mul_right s.n (Nat.succ_le_of_lt hi.1)
  nlinarith [hi.2]

/-- Scalar typed geometry is reused only to prove ownership of the actual
quoted store offsets; it supplies no assumed execution relation. -/
theorem grid_writes_unique (layout : Layout) (s : Shape) (c : Fin 15) (initial : State Word32)
    (hout : OutputLayout layout s) (p q : Nat) (hp : p < s.programs c) (hq : q < s.programs c)
    (loc : RegionName × Nat) (wp : tileWrites layout s c initial p loc)
    (wq : tileWrites layout s c initial q loc) : p = q := by
  let geometry : Buffers := ⟨0,0,0,0,0,0,0,0,0⟩
  have ownership (pid : Nat) (write : tileWrites layout s c initial pid loc) :
      Whole001.writes s geometry c pid (geometry.outRegion,loc.2) := by
    obtain ⟨i,hi,he⟩ := write
    have off := congrArg Prod.snd he
    rw [output_coordinates_source layout s c (tileState initial pid) i hout,
      source_row,source_column] at off
    rw [off]
    exact source_store_ownership s geometry c pid i.1.val i.2.1.val i.1.isLt i.2.1.isLt hi.1 hi.2
  exact Whole001.writes_unique s geometry c p q hp hq _ (ownership p wp) (ownership q wq)

def gridMemory (layout : Layout) (s : Shape) (c : Fin 15) (initial : State Word32) : Backing :=
  Schedule.run (frozenStep layout s c initial) (List.range (s.programs c)) (flatten initial)

theorem grid_values (layout : Layout) (s : Shape) (c : Fin 15) (initial : State Word32)
    (hout : OutputLayout layout s) (m n : Nat) (hm : m < s.m) (hn : n < s.n) :
    gridMemory layout s c initial (layout.output,Whole001.outputOffset s m n) =
      int32Cell (matrixWordTarget layout s initial m n) := by
  unfold gridMemory frozenStep
  rw [fixed_cell, if_pos (grid_covered layout s c initial hout m n hm hn)]
  have coords := Whole001.output_coordinates s m n hn
  simp only [denseCell,Prod.snd,coords.1,coords.2]

theorem grid_frame (layout : Layout) (s : Shape) (c : Fin 15) (initial : State Word32)
    (hout : OutputLayout layout s) (loc : RegionName × Nat)
    (outside : loc.1 ≠ layout.output ∨ s.m*s.n ≤ loc.2) :
    gridMemory layout s c initial loc = flatten initial loc := by
  unfold gridMemory frozenStep
  apply Schedule.fixed_frame
  intro pid hp write
  have bounds := writes_inside layout s c initial hout pid loc write
  rcases outside with hr | ho
  · exact hr bounds.1
  · omega

theorem grid_low32 (cast : Word32 → Byte)
    (castLeaf : ∀ word, DefinedCast word → cast word = referenceCast word)
    (rest : Algebra Word32) (layout : Layout) (s : Shape) (c : Fin 15) (legal : s.legal c)
    (initial : State Word32) (defined : FP32Domain layout s initial) (bytes : UInt8Domain layout s initial)
    (hout : OutputLayout layout s) (separate : layout.output ≠ layout.input ∧ layout.output ≠ layout.weight) :
    GridExec cast rest (parameters layout s c) initial (List.range (s.programs c))
      (flatten initial) (gridMemory layout s c initial) := by
  by_cases empty : s.m = 0 ∨ s.n = 0
  · have gridZero : s.programs c = 0 := by
      have bm : 0 < (configs c).bm := by have := config_bounds c; omega
      have bn : 0 < (configs c).bn := by have := config_bounds c; omega
      rcases empty with hm | hn
      · simp [Shape.programs,Shape.rows,hm,ceil_zero (configs c).bm bm]
      · simp [Shape.programs,Shape.cols,hn,ceil_zero (configs c).bn bn]
    simp only [gridZero,List.range_zero,gridMemory,Schedule.run,List.foldl_nil]
    exact GridExec.nil _
  · exact schedule_execution cast castLeaf rest layout s c legal initial (by omega) (by omega)
      defined bytes hout separate _ _ (by intro r o hr; rfl)

#axiomsClean fixed_cell
#axiomsClean grid_covered
#axiomsClean writes_inside
#axiomsClean grid_writes_unique
#axiomsClean grid_values
#axiomsClean grid_frame
#axiomsClean grid_low32
end
end VeriTile.Bench.Repairs.Int8MatmulKernelModularChecked.TileModel
