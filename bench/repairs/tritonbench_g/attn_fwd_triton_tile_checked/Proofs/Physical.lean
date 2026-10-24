import bench.repairs.tritonbench_g.attn_fwd_triton_tile_checked.Proofs.Execution
import bench.repairs.tritonbench_g.attn_fwd_triton.Proofs.NativeReturn
import bench.repairs.tritonbench_g.attn_fwd_triton_tile_checked.Contract

namespace VeriTile.Bench.Repairs.AttnFwdTritonTileChecked
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open AttnFwdTriton AttnFwdTriton.TypedExecution
open VeriTile.Bench.Repairs.LayerNormFwd
open TypedPhysical (Physical pull)
open Verified.PhysicalCells (Layout Role FreshBytes)
open Graph
noncomputable section

namespace Physical

def liftValue : (d : TileDType) → Value Word d → Value Expr d
  | .real | .fp32 | .fp16 | .bf16 | .f8e4 | .f8e5 => Expr.leaf
  | .nat | .int | .bool | .ptr | .blockPtr => id

def evaluateValue (ops : Operations) : (d : TileDType) → Value Expr d → Value Word d
  | .real | .fp32 | .fp16 | .bf16 | .f8e4 | .f8e5 => evaluate ops
  | .nat | .int | .bool | .ptr | .blockPtr => id

def liftCell : Cell Word → Cell Expr | .mk d v => .mk d (liftValue d v)
def evaluateCell (ops : Operations) : Cell Expr → Cell Word | .mk d v => .mk d (evaluateValue ops d v)

def liftPhysical {a} (memory : Physical Word a) : Physical Expr a := fun cell => liftCell (memory cell)
def evaluatePhysical (ops : Operations) {a} (memory : Physical Expr a) : Physical Word a :=
  fun cell=>evaluateCell ops (memory cell)

def liftState (s : State Word) : State Expr where
  mem := fun r o=>liftCell (s.mem r o)
  regs := fun d sh n=>(s.regs d sh n).map (fun xs i=>liftValue d (xs i))
  pids := s.pids
  numPids := s.numPids
  undef := fun d r o=>liftValue d (s.undef d r o)

@[simp] theorem evaluate_lift (ops : Operations) (c : Cell Word) : evaluateCell ops (liftCell c)=c := by
  cases c with | mk d v=>cases d <;> rfl

/-- FP32 compute loads in Structural use the erased `.real` cell channel.
The payload here is still the exact 32 input bits, without numerical decoding. -/
def callerCells (a : Layout) (memory : Checked.ByteMemory) : Physical Word a.toLayout :=
  fun cell=>Cell.mk .real (Checked.readFP32 memory cell.val)

def initial (a : Layout) (memory : Checked.ByteMemory) (template : State Word) : State Expr :=
  pull a.toLayout (liftPhysical (callerCells a memory)) (liftState template)

def reference (ops : Operations) (p : Config) (a : Layout)
    (memory : Checked.ByteMemory) (template : State Word) (c : Coordinate p) : Cell Word :=
  evaluateCell ops (Model.reference p (HostShape.blockD p.headDim) (HostShape.blockD_covers _)
    (initial a memory template).mem c)

theorem actual_input (p : Config) (a : Layout) (objects : Role→Checked.TensorDescriptor)
    (views : Views a p (Checked.pointers objects)) (memory : Checked.ByteMemory)
    (template : State Word)
    (hinput : Checked.inputChecks (Checked.typedGeometry p) (objects .query).metadata
      (objects .key).metadata (objects .value).metadata (objects .queryScale).metadata
      (objects .keyScale).metadata=Bool.true)
    (z h n d : Nat) (r : Role) (hr : r≠Role.output) (ha : roleActive p z h n d r) :
    ((initial a memory template).mem (roleKey p z h n d r).1 (roleKey p z h n d r).2).read .real=
      Expr.leaf (Checked.descriptorRead objects memory r z h n d) := by
  have hl:=views.live z h n d r ha
  have hp:=views.address z h n d r ha
  simp only [initial,pull,TypedPhysical.read,dif_pos hl,liftPhysical,callerCells,liftCell,liftValue,
    Cell.read_same]
  rw [hp]
  simp only [Checked.inputChecks,Bool.and_eq_true,decide_eq_true_eq] at hinput
  have hq:=hinput.2.2.2.2.1
  have hk:=hinput.2.2.2.2.2.1
  have hv:=hinput.2.2.2.2.2.2
  cases r <;> try contradiction
  all_goals simp [Checked.descriptorRead,Checked.descriptorElement,Checked.pointers,roleOffset,
    Verified.PhysicalCells.width,hq,hk,hv,Checked.typedGeometry,Checked.Strides.raw]

theorem grid (ops : Operations) (p : Config) (a : Layout) (base : Role→Nat)
    (views : Views a p base) (memory : Checked.ByteMemory) (template : State Word)
    (stage : p.stage=3) (hn : NonOverlapping p)
    (fresh : FreshBytes a (fun k=>Window p k.1 k.2)) :
    ∃ final, ∃ graph : Physical Expr a.toLayout, ∃ physical : Physical Word a.toLayout,
      Nonempty (GridRun backend p (HostShape.blockD p.headDim) Bool.false (initial a memory template) final) ∧
      pull a.toLayout graph final=final ∧ physical=evaluatePhysical ops graph ∧
      (∀ c : Coordinate p,physical (outputCell a p base views c)=reference ops p a memory template c) ∧
      (∀ cell : Fin a.capacity,(∀ k,Window p k.1 k.2 → a.address k≠cell.val) →
        physical cell=callerCells a memory cell) := by
  obtain ⟨final,graph,hg,hpull,hread,hframe⟩ := physical_division_grid backend p
    (HostShape.blockD p.headDim) (support _) a base views
    (liftPhysical (callerCells a memory)) (liftState template) (HostShape.blockD_covers _) hn fresh
  refine ⟨final,graph,evaluatePhysical ops graph,hg,hpull,rfl,?_,?_⟩
  · intro c
    change evaluateCell ops (graph (outputCell a p base views c))=_
    rw [hread c,Model.reference_eq p _ stage hn (HostShape.blockD_covers _)]
    rfl
  · intro cell hc
    change evaluateCell ops (graph cell)=_
    rw [hframe cell hc]
    exact evaluate_lift ops _

#specNonCircular reference avoiding [FP.Structural.exec,FP.Structural.run,TypedExecution.reference]
#axiomsClean actual_input
#axiomsClean grid
end Physical
end
end VeriTile.Bench.Repairs.AttnFwdTritonTileChecked
