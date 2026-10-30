import bench.repairs.tritonbench_g.rotary_emb.Proofs.TypedExecution
import bench.repairs.tritonbench_g.rotary_emb.Proofs.Geometry

namespace VeriTile.Bench.Repairs.RotaryEmb.Typed
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.RotaryEmbNopad
noncomputable section

/-- The exact fixed launch geometry and checked contiguous source strides. -/
def layout (shape : Shape) : Kernels.Layout where
  Q := "Q"
  K := "K"
  Cos := "Cos"
  Sin := "Sin"
  stride_qbs := shape.headsQ*shape.width
  stride_qh := shape.width
  stride_qd := 1
  stride_kbs := shape.headsK*shape.width
  stride_kh := shape.width
  stride_kd := 1
  stride_cosbs := shape.cosWidth
  stride_cosd := 1
  stride_sinbs := shape.sinWidth
  stride_sind := 1
  max_total_len := shape.tokens
  ROTARY_DIM := shape.rotary
  HEAD_Q := shape.headsQ
  HEAD_K := shape.headsK
  BLOCK_HEAD := 4
  BLOCK_SEQ := 16
  BLOCK_PAIRS := shape.pairBlock

def work {A : Type} (shape : Shape) (s : State A) (key : Bool)
    (i : Lane (layout shape)) : Work :=
  ⟨key,s.pids 1*16+i.1.val,s.pids 0*4+i.2.1.val,i.2.2.1.val⟩

theorem offset_cell {A : Type} (shape : Shape) (s : State A) (key odd : Bool)
    (i : Lane (layout shape)) :
    offset (layout shape) s key odd i=shape.cell (work shape s key i) odd := by
  cases key <;> cases odd <;>
    simp [offset,layout,token,head,dimension,work,Shape.cell,Shape.address,
      Shape.heads,Nat.add_mul,Nat.mul_assoc,Nat.mul_comm] <;> ring

theorem active_work {A : Type} (shape : Shape) (valid : shape.Valid) (s : State A)
    (key : Bool) (i : Lane (layout shape)) :
    active (layout shape) s key i ↔ shape.Active (work shape s key i) := by
  have even := even_width shape valid
  cases key <;>
    simp only [active,layout,token,head,dimension,work,Shape.Active,Shape.heads,
      Bool.false_eq_true,↓reduceIte]
  all_goals omega

theorem work_injective {A : Type} (shape : Shape) (s : State A) (key : Bool) :
    Function.Injective (work shape s key) := by
  intro i j same
  have row := congrArg Work.token same
  have hd := congrArg Work.head same
  have pair := congrArg Work.pair same
  simp only [work] at row hd pair
  exact Prod.ext (Fin.ext (by omega))
    (Prod.ext (Fin.ext (by omega)) (Prod.ext (Fin.ext pair) (Subsingleton.elim _ _)))

/-- Existing logical address injectivity discharges both halves' typed store
ownership. No distinct-store or final-output fact is left as a premise. -/
theorem writes_apart {A : Type} (shape : Shape) (valid : shape.Valid) (s : State A) (key : Bool) :
    Pair.WritesApart (active (layout shape) s key) (offset (layout shape) s key) := by
  constructor
  · intro odd i j hi hj eq
    have same := cell_injective shape valid (work shape s key i) (work shape s key j) odd odd
      ((active_work shape valid s key i).mp hi) ((active_work shape valid s key j).mp hj) rfl
      (by simpa only [offset_cell] using eq)
    exact work_injective shape s key same.1
  · intro i j hi hj eq
    have same := cell_injective shape valid (work shape s key i) (work shape s key j) Bool.false Bool.true
      ((active_work shape valid s key i).mp hi) ((active_work shape valid s key j).mp hj) rfl
      (by simpa only [offset_cell] using eq)
    cases same.2

#axiomsClean offset_cell
#axiomsClean active_work
#axiomsClean writes_apart
end
end VeriTile.Bench.Repairs.RotaryEmb.Typed
