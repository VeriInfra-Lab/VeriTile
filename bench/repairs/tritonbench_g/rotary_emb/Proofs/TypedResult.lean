import bench.repairs.tritonbench_g.rotary_emb.Proofs.Typed
import bench.repairs.tritonbench_g.rotary_emb.Proofs.TypedGeometry
import bench.repairs.tritonbench_g.rotary_emb.Spec

namespace VeriTile.Bench.Repairs.RotaryEmb.Typed
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.RotaryEmbNopad
noncomputable section
set_option maxHeartbeats 2000000

@[simp] theorem branch_pids {A : Type} [Inhabited A] (ops : Ops A) (p : Kernels.Layout)
    (s : State A) (key : Bool) (entry : State A) :
    (branch ops p s key entry).pids=entry.pids := by
  simp [branch,Pair.written,Pair.halfScatter]

theorem branch_other {A : Type} [Inhabited A] (ops : Ops A) (p : Kernels.Layout)
    (s : State A) (key : Bool) (entry : State A) (r : RegionName) (o : Nat)
    (other : r≠region p key) : (branch ops p s key entry).mem r o=entry.mem r o :=
  Pair.frame _ _ _ _ _ _ _ (Or.inl other)

@[simp] theorem value_afterQ {A : Type} [Inhabited A] (ops : Ops A)
    (shape : Shape) (s : State A) (odd : Bool) (i : Lane (layout shape)) :
    value ops (layout shape) (branch ops (layout shape) s Bool.false s) Bool.true odd i=
      value ops (layout shape) s Bool.true odd i := by
  simp only [value,offset,coefficient,token,head,branch_pids]
  simp [branch_other,region,layout]

/-- Independent simultaneous rotation of the two disjoint payload regions.
Every value is read from the original memory, and all other cells are kept. -/
def target {A : Type} [Inhabited A] (ops : Ops A) (shape : Shape) (s : State A) : State A :=
  branch ops (layout shape) s Bool.true (branch ops (layout shape) s Bool.false s)

theorem reference_eq_target {A : Type} [Inhabited A] (ops : Ops A)
    (shape : Shape) (s : State A) : reference ops (layout shape) s=target ops shape s := by
  unfold reference target
  let q := branch ops (layout shape) s Bool.false s
  have ha : active (layout shape) q Bool.true=active (layout shape) s Bool.true := by
    funext i
    simp [active,token,head,q]
  have ho : offset (layout shape) q Bool.true=offset (layout shape) s Bool.true := by
    funext odd i
    simp [offset,token,head,q]
  have hv : value ops (layout shape) q Bool.true=value ops (layout shape) s Bool.true := by
    funext odd i
    exact value_afterQ ops shape s odd i
  change Pair.written _ (active (layout shape) q Bool.true)
    (offset (layout shape) q Bool.true) (value ops (layout shape) q Bool.true) q=
    Pair.written _ (active (layout shape) s Bool.true)
      (offset (layout shape) s Bool.true) (value ops (layout shape) s Bool.true) q
  simp only [ha,ho,hv]

def Outside {A : Type} (shape : Shape) (s : State A) (r : RegionName) (o : Nat) : Prop :=
  ∀ key,Pair.Outside (region (layout shape) key) (active (layout shape) s key)
    (offset (layout shape) s key) r o

theorem target_frame {A : Type} [Inhabited A] (ops : Ops A) (shape : Shape)
    (s : State A) (r : RegionName) (o : Nat) (outside : Outside shape s r o) :
    (target ops shape s).mem r o=s.mem r o := by
  unfold target branch
  rw [Pair.frame _ _ _ _ _ _ _ (outside Bool.true),
    Pair.frame _ _ _ _ _ _ _ (outside Bool.false)]

theorem target_readback {A : Type} [Inhabited A] (ops : Ops A) (shape : Shape)
    (valid : shape.Valid) (s : State A) (key odd : Bool) (i : Lane (layout shape))
    (live : active (layout shape) s key i) :
    (target ops shape s).mem (region (layout shape) key) (offset (layout shape) s key odd i)=
      value ops (layout shape) s key odd i := by
  cases key
  · unfold target
    rw [branch_other _ _ _ _ _ _ _ (by simp [region,layout])]
    exact Pair.readback _ _ _ _ _ (writes_apart shape valid s Bool.false) odd i live
  · exact Pair.readback _ _ _ _ _ (writes_apart shape valid s Bool.true) odd i live

theorem successful {A : Type} [Inhabited A] (rest : Algebra A) (ops : Ops A)
    (shape : Shape) (valid : shape.Valid) (s : State A) :
    ∃ t,exec (algebra rest ops) (Kernels.kernel (layout shape)) s=some t ∧
      t.mem=(target ops shape s).mem ∧
      (∀ key odd i,active (layout shape) s key i →
        t.mem (region (layout shape) key) (offset (layout shape) s key odd i)=
          value ops (layout shape) s key odd i) ∧
      (∀ r o,Outside shape s r o → t.mem r o=s.mem r o) := by
  refine ⟨(execution rest ops (layout shape) s).val,(execution rest ops (layout shape) s).property,?_,?_,?_⟩
  all_goals rw [execution_memory,reference_eq_target]
  · exact target_readback ops shape valid s
  · exact target_frame ops shape s

def name : Region → RegionName
  | .q => "Q"
  | .k => "K"
  | .cos => "Cos"
  | .sin => "Sin"
  | .other n => ⟨"other"++toString n⟩

def inputs {A : Type} [Inhabited A] (s : State A) : Memory A := fun r o =>
  (s.mem (name r) o).read .real

theorem value_formula (shape : Shape) (s : State ℝ) (key odd : Bool)
    (i : Lane (layout shape)) :
    value realOps (layout shape) s key odd i=.mk .real
      (formula shape (inputs s) key (token (layout shape) s i) (head (layout shape) s i)
        (dimension (layout shape) odd i)) := by
  have h := pair_formula shape (inputs s) (work shape s key i) odd
  have hv : value realOps (layout shape) s key odd i=
      Cell.mk .real (component (pairValues shape realOps (work shape s key i) (inputs s)) odd) := by
    simp only [value,offset_cell]
    cases key <;> cases odd <;>
      simp [coefficient,layout,dimension,token,head,work,inputs,name,
        component,pairValues,sourcePair,realOps,Shape.cell,Shape.coefficient,region,dataRegion,
        Nat.mul_comm]
  rw [hv,h]
  cases odd <;> simp [dimension,work,token,head,layout,Nat.mul_comm]

#specNonCircular target avoiding [Kernels.kernel,execution,reference]
#axiomsClean successful
#axiomsClean value_formula
end
end VeriTile.Bench.Repairs.RotaryEmb.Typed
