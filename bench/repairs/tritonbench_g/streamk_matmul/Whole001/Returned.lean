import bench.repairs.tritonbench_g.streamk_matmul.Whole001.CTASchedule
namespace VeriTile.Bench.Repairs.StreamK.Whole001
namespace Returned
open Source Plan
set_option maxHeartbeats 16000000
noncomputable section

abbrev Cell (p : Parameters) := Fin p.h.m × Fin p.h.n

structure Owner (p : Parameters) (cell : Cell p) where
  tile : Nat
  bound : tile < tiles p.h
  lane : Lane p
  row_eq : row p tile lane=cell.1.val
  col_eq : col p tile lane=cell.2.val

theorem owner_exists (p : Parameters) (cell : Cell p) : Nonempty (Owner p cell) := by
  obtain ⟨tile,⟨ht,⟨lane,hr,hc⟩⟩,hu⟩:=Geometry.output_owner p.h p.blockM p.blockN cell
  exact ⟨⟨tile,ht,(lane.1,lane.2.1),hr,hc⟩⟩

def owner (p : Parameters) (cell : Cell p) : Owner p cell := Classical.choice (owner_exists p cell)

def index (p : Parameters) (cell : Cell p) : C p :=
  ⟨cell.1.val*p.h.n+cell.2.val,Geometry.packed_bound _ _ _ _ cell.1.isLt cell.2.isLt⟩

structure Tensor (α : Type) (p : Parameters) where
  storage : C p → α
  -- The wrapper creates a fresh contiguous torch.empty((M,N),dtype=FP32).
  shape : Nat × Nat := (p.h.m,p.h.n)
  strides : Nat × Nat := (max p.h.n 1,1)
  shape_eq : shape=(p.h.m,p.h.n) := by rfl
  strides_eq : strides=(max p.h.n 1,1) := by rfl

def read (p : Parameters) (tensor : Tensor α p) (cell : Cell p) : α := tensor.storage (index p cell)

def tensor (p : Parameters) (ops : Instructions α) (initial : State α p exterior) : Tensor α p :=
  ⟨(Source.call p ops initial).output,(p.h.m,p.h.n),(max p.h.n 1,1),rfl,rfl⟩

def value (p : Parameters) (ops : Instructions α) (a : A p → α) (b : B p → α) (cell : Cell p) : α :=
  let o:=owner p cell
  if ht:o.tile < (select p.h).streamTiles then Reference.streamValue p ops a b o.tile ht o.lane
  else ops.halfRoundTrip (chunkValue p ops a b o.tile 0 (select p.h).iterations o.lane)

/-- Public returned-buffer theorem for every element of the actual M*N
allocation. No successful execution or output equation is a premise. -/
theorem value_contract (p : Parameters) (ops : Instructions α) (initial : State α p exterior)
    (cell : Cell p) : read p (tensor p ops initial) cell=value p ops initial.inputA initial.inputB cell := by
  let o:=owner p cell
  have ha:active p o.tile o.lane:=by
    constructor
    · rw [o.row_eq]; exact cell.1.isLt
    · rw [o.col_eq]; exact cell.2.isLt
  have hi:output p o.tile o.lane ha=index p cell:=by
    apply Fin.ext
    simp only [output,index,o.row_eq,o.col_eq]
  change (Source.call p ops initial).output (index p cell)=_
  rw [←hi]
  unfold value
  dsimp only
  split
  next ht => exact call_stream_value p ops initial o.tile ht o.lane ha
  next ht => exact call_full_value p ops initial o.tile o.bound (by change ¬o.tile < (select p.h).streamTiles at ht; omega) o.lane ha

/-- Complete ideal-Real mathematical contract. FP16 casts are identities in
this interpretation; TF32 instructions are interpreted as exact local MMA. -/
theorem real_contract (p : Parameters) (initial : State ℝ p exterior) (cell : Cell p) :
    read p (tensor p Arithmetic.real initial) cell=Reference.matrixProduct p initial.inputA initial.inputB cell.1 cell.2 := by
  rw [value_contract]
  unfold value
  let o:=owner p cell
  have ha:active p o.tile o.lane:=by
    constructor
    · rw [o.row_eq]; exact cell.1.isLt
    · rw [o.col_eq]; exact cell.2.isLt
  have hrow:(⟨row p o.tile o.lane,ha.1⟩ : Fin p.h.m)=cell.1:=Fin.ext o.row_eq
  have hcol:(⟨col p o.tile o.lane,ha.2⟩ : Fin p.h.n)=cell.2:=Fin.ext o.col_eq
  dsimp only
  split
  next ht =>
    rw [Reassembly.stream_real p initial.inputA initial.inputB o.tile ht o.lane ha,hrow,hcol]
  next ht =>
    change chunkValue p Arithmetic.real initial.inputA initial.inputB o.tile 0 (select p.h).iterations o.lane=_
    rw [Arithmetic.full_real p initial.inputA initial.inputB o.tile o.lane ha,hrow,hcol]

/-- Same exact returned value for every complete source CTA interleaving. -/
theorem scheduled_contract (p : Parameters) (ops : Instructions α) (initial : State α p exterior)
    (first : Schedule.Order (firstWave p ops).length) (reduce : Schedule.CTAInterleaving p)
    (last : Schedule.Order (fullWave p ops).length) (cell : Cell p) :
    (Schedule.call p ops first reduce last initial).output (index p cell)=
      value p ops initial.inputA initial.inputB cell := by
  rw [Schedule.call_eq]
  exact value_contract p ops initial cell

#axiomsClean value_contract
#axiomsClean real_contract
#axiomsClean scheduled_contract
#specNonCircular value avoiding [Source.call,Schedule.call,Memory.run,Memory.step,tensor]
end
end Returned
end VeriTile.Bench.Repairs.StreamK.Whole001
