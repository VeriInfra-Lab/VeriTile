import bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.Geometry

namespace VeriTile.Bench.Repairs.AttentionScoreTileChecked.Reused.DenseSource
open VeriTile.Triton Geometry
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 16000

def programState (c : Shape) (s : BlockState) (block z h : Nat) : BlockState :=
  { s with pids := fun dim => if dim = 0 then block else if dim = 1 then z*c.H+h else 0 }

def inputs (c : Shape) (s : BlockState) (z h : Nat) : RealFormula.Inputs c.D :=
  RealLoads.inputs s "q" "k" "m" c.D (c.queryBase z h) (c.keyBase z h)
    (c.normalizerBase z h) c.D 1 c.D 1

def kernel (c : Shape) (offset size : Int) (scale : ℝ) (window complement : Bool) : ComputeKernel :=
  RealSource.source c.B c.D (parameters c offset size scale window complement) Bool.false

def config (c : Shape) (block z h : Nat) (offset size : Int) (scale : ℝ)
    (window complement : Bool) : RealLoop.Config where
  Q := "q"
  K := "k"
  M := "m"
  B := c.B
  D := c.D
  N := c.N
  NK := c.NK
  baseQ := c.queryBase z h
  baseK := c.keyBase z h
  baseM := c.normalizerBase z h
  sqm := c.D
  sqk := 1
  skn := c.D
  skk := 1
  startN := block
  offset := offset
  size := size
  scale := scale
  window := window
  complement := complement
  evenM := decide (c.N % c.B = 0)
  evenN := decide (c.NK % c.B = 0)
  candidate := Bool.false

theorem batch_head (c : Shape) (z : Nat) (h : Fin c.H) :
    (z*c.H+h.val)/c.H = z ∧ (z*c.H+h.val)%c.H = h.val := by
  have hh : 0 < c.H := Nat.zero_lt_of_lt h.isLt
  have ho := owner_unique c.H (z*c.H+h.val) z h.val hh h.isLt rfl
  exact ⟨ho.1.symm,ho.2.symm⟩

theorem config_eq (c : Shape) (s : BlockState) (block z : Nat) (h : Fin c.H)
    (offset size : Int) (scale : ℝ) (window complement : Bool) :
    RealSource.config c.B c.D (parameters c offset size scale window complement) Bool.false
      (programState c s block z h.val) = config c block z h.val offset size scale window complement := by
  obtain ⟨hz,hh⟩ := batch_head c z h
  simp [RealSource.config, RealPrelude.config, parameters, programState, config,
    hz, hh, Shape.queryBase, Shape.keyBase, Shape.normalizerBase, Nat.add_mul, Nat.mul_assoc]

theorem output_base (c : Shape) (z : Nat) (h : Fin c.H)
    (offset size : Int) (scale : ℝ) (window complement : Bool) :
    RealSource.outputBase (parameters c offset size scale window complement) (z*c.H+h.val) =
      c.outputBase z h.val := by
  obtain ⟨hz,hh⟩ := batch_head c z h
  simp [RealSource.outputBase, parameters, hz, hh, Shape.outputBase, Nat.add_mul, Nat.mul_assoc]

/-- The complete repaired source runs from initial memory and returns its
independent column value at every active lane. No execution/output premise is
required; only the actual host metadata is instantiated. -/
theorem executes (c : Shape) (s : BlockState) (block z : Nat) (h : Fin c.H)
    (offset size : Int) (scale : ℝ) (window complement : Bool)
    (keyOffsetBound : block*c.B < 2147483648) :
    ∃ t, exec (kernel c offset size scale window complement).eraseDType.toAlgKernel
        (programState c s block z h.val) = some t ∧
      (∀ lane : Fin c.B, block*c.B+lane.val < c.NK →
        t.readMem "out" (c.outputBase z h.val+block*c.B+lane.val) =
          LogicalFormula.column (inputs c s z h.val) scale c.N window complement offset size
            (block*c.B+lane.val)) ∧
      (∀ (r : RegionName) a, (r ≠ "out" ∨ ∀ lane : Fin c.B, block*c.B+lane.val < c.NK →
        a ≠ c.outputBase z h.val+block*c.B+lane.val) → t.mem r a = s.mem r a) := by
  obtain ⟨t,ht,ho,hf⟩ := RealSource.executes (programState c s block z h.val) c.B c.D
    (parameters c offset size scale window complement) Bool.false (supported_positive c.B c.block)
    (by simpa [programState] using keyOffsetBound)
  refine ⟨t,ht,?_,?_⟩
  · intro lane hl
    obtain ⟨hz,hh⟩ := batch_head c z h
    have hv : t.readMem "out" (c.outputBase z h.val+block*c.B+lane.val) =
        RealFormula.column (inputs c s z h.val) scale c.N c.NK c.B (RealLoop.tripCount c.N c.B)
          window complement (decide (c.N % c.B = 0)) (decide (c.NK % c.B = 0)) offset size
          (block*c.B+lane.val) := by
      simpa [RealSource.outputBase, RealSource.config, RealSource.blocks,
        RealPrelude.config, RealLoop.Config.column, RealLoop.Config.inputs,
        inputs, parameters, programState, RealLoads.inputs, Shape.outputBase,
        Shape.queryBase, Shape.keyBase, Shape.normalizerBase, hz, hh, hl,
        Nat.add_mul, Nat.mul_assoc, Nat.add_assoc] using ho lane
    exact hv.trans (LogicalFormula.repaired_column (inputs c s z h.val) scale c.N c.NK c.B
      (block*c.B+lane.val) window complement _ _ offset size
      (supported_positive c.B c.block) hl (by simp))
  · intro r a ha
    apply hf r a
    change (r ≠ ("out" : RegionName) ∨ ∀ lane : Fin c.B, block*c.B+lane.val<c.NK →
      a ≠ RealSource.outputBase (parameters c offset size scale window complement)
        (z*c.H+h.val)+block*c.B+lane.val)
    rw [output_base]
    exact ha

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttentionScoreTileChecked.Reused.DenseSource
