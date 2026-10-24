import bench.repairs.tritonbench_g.decay_cumsum.Proofs.KernelCompiler
import bench.repairs.tritonbench_g.decay_cumsum.Proofs.KernelReal
import bench.repairs.tritonbench_g.fast_layernorm.Proofs.KernelStorage

/-! The actual packed kernel arguments, including the interior tail pointer.
No allocator or host outcome is an argument. Backward dqo/dko are the same
input/output Tensor objects. Logical reads use their entry values. -/
namespace VeriTile.Bench.Repairs.DecayCumsum.KernelTensor
open VeriTile.Triton FP.Structural Geometry VeriTile.Torch.FP32
namespace Shared
export VeriTile.Bench.Repairs.FastLayernorm.KernelStorage
  (Placement TensorImage commit complete native_logical_output)
end Shared
open VeriTile.Bench.Repairs.LayernormFwdTriton
noncomputable section
open scoped Classical

structure Buffers (h : Call) (kind : Kind) where
  tensor : RegionName → Tensor
  laws : ∀ r,TensorLaws (tensor r)
  checked : ∀ r,VeriTile.Bench.Repairs.FastLayernorm.Contract.TensorContract (tensor r).metadata Bool.true
  shape : ∀ r∈argumentRoles kind,(tensor r).metadata.shape=[h.B,h.H,h.T,h.width]
  device : ∀ r∈argumentRoles kind,(tensor r).metadata.device=(tensor (firstRole kind)).metadata.device

def Buffers.view (b : Buffers h kind) (r : RegionName) : DataView :=
  dataView (b.tensor r) (b.laws r) ((b.checked r).dense rfl)
theorem Buffers.count (b : Buffers h kind) (r : RegionName) (hr : r∈argumentRoles kind) :
    (b.tensor r).metadata.numel=h.cells := by
  simp [VeriTile.Torch.TensorMetadata.numel,b.shape r hr,Call.cells,Call.rows,Call.headStride,Nat.mul_assoc]
def shift (h : Call) (p : Phase) := p.start*h.width
def sourcePointer (b : Buffers h kind) (p : Phase) (r : RegionName) : BitVec 64 :=
  BitVec.ofNat 64 ((b.view r).address (shift h p))
theorem original_pointer (b : Buffers h kind) (r : RegionName)
    (nonempty : 0<(b.tensor r).metadata.numel) :
    BitVec.ofNat 64 ((b.view r).address 0)=(b.tensor r).dataPtr :=
  actual_pointer (b.tensor r) (b.laws r) ((b.checked r).dense rfl) nonempty
theorem tail_pointer (b : Buffers h kind) (p : Phase) (r : RegionName)
    (nonempty : 0<(b.tensor r).metadata.numel) :
    sourcePointer b p r=(b.tensor r).dataPtr+BitVec.ofNat 64 (4*shift h p) := by
  rw [←original_pointer b r nonempty]
  simp only [sourcePointer,DataView.address,Nat.mul_zero,Nat.add_zero,←BitVec.ofNat_add]

variable {α : Type} [Inhabited α]
def physical (b : Buffers h kind) (p : Phase) (read : Nat → α) (u : State α) : State α :=
  {u with mem:=fun r i => if shift h p+i<(b.tensor r).metadata.numel then
    (Cell.mk .real (read ((b.view r).address (shift h p+i)))) else u.mem r i}
def logical (b : Buffers h kind) (p : Phase) (read : Nat → α) (u : State α) : State α :=
  {u with mem:=fun r i => if shift h p+i<(b.tensor r).metadata.numel then
    (Cell.mk .real (read ((b.tensor r).storage.address ((b.tensor r).physicalIndex (shift h p+i))))) else u.mem r i}
theorem logical_address (b : Buffers h kind) (r : RegionName) (i : Nat)
    (hi : i<(b.tensor r).metadata.numel) :
    (b.tensor r).storage.address ((b.tensor r).physicalIndex i)=(b.view r).address i := by
  rw [(b.laws r).contiguous ((b.checked r).dense rfl) i hi]
  simp only [Buffers.view,dataView,DataView.address,DataView.base,
    if_neg (by omega : (b.tensor r).metadata.numel≠0),Storage.address]
  omega
theorem physical_eq_logical (b : Buffers h kind) (p : Phase) (read : Nat → α) (u : State α) :
    physical b p read u=logical b p read u := by
  have mem : (physical b p read u).mem=(logical b p read u).mem := by
    funext r i
    by_cases hi : shift h p+i<(b.tensor r).metadata.numel
    · simp [physical,logical,hi,logical_address b r _ hi]
    · simp [physical,logical,hi]
  exact congrArg (fun m => ({u with mem:=m} : State α)) mem

theorem shifted (h : Call) (p : Phase) (prog : Program h p) (t : Nat) (j : Fin h.BK) :
    shift h p+FPPrefix.off h p prog t j=Geometry.offset h p prog t j.val := by
  simp only [shift,FPPrefix.off,Geometry.offset,Geometry.time,Geometry.column,Call.headStride]
  ring
theorem coverage (b : Buffers h kind) (p : Phase) (hp : p∈phases h) (prog : Program h p)
    (r : RegionName) (hr : r∈argumentRoles kind) (t : Fin p.span) (j : Fin h.BK) :
    shift h p+FPPrefix.off h p prog t.val j<(b.tensor r).metadata.numel := by
  rw [shifted,b.count r hr]
  exact offset_inside h p hp prog t j

theorem native_input (b : Buffers h kind) (p : Phase) (hp : p∈phases h) (prog : Program h p)
    (read : Nat → Native.Word) (u : State Native.Node) (r : RegionName)
    (hr : r∈argumentRoles kind) (t : Fin p.span) (j : Fin h.BK) :
    (KernelExecution.read h p prog (physical b p (fun a=>Native.atom (read a)) u) j r t.val).value=
      (logicalValue (fun a => (⟨read a⟩ : Word)) (b.tensor r) (b.laws r)
        (Geometry.offset h p prog t.val j.val)).bits := by
  have hi := coverage b p hp prog r hr t j
  simp only [KernelExecution.read,physical,hi,ite_true,Cell.read_same,Native.atom]
  rw [shifted]
  simp only [logicalValue,(b.checked r).noNeg,Bool.false_eq_true,if_false]
  rw [logical_address b r _ (by rw [←shifted]; exact hi)]

def active (kind : Kind) (h : Call) (p : Phase) (prog : Program h p) (r : RegionName) (i : Nat) : Prop :=
  r∈FPTile.outputs kind ∧ ∃ t : Fin p.span,∃ j : Fin h.BK,i=Geometry.offset h p prog t.val j.val
def relocated (h : Call) (p : Phase) (v : State α) : State α :=
  {v with mem:=fun r i => v.mem r (i-shift h p)}
theorem relocated_output (h : Call) (p : Phase) (prog : Program h p) (v : State α)
    (r : RegionName) (t : Fin p.span) (j : Fin h.BK) :
    (relocated h p v).mem r (Geometry.offset h p prog t.val j.val)=v.mem r (FPPrefix.off h p prog t.val j) := by
  rw [←shifted]
  simp [relocated]

/-- Address metadata only. This identifies output readback with the same
Tensor arguments that supplied the kernel pointers, including in-place roles. -/
structure Outputs (b : Buffers h kind) (p : Phase) (prog : Program h p) extends
    Shared.Placement (active kind h p prog) where
  same : ∀ r∈FPTile.outputs kind,tensor r=b.tensor r

/-- Distinct scratch roles that a source store could otherwise overwrite.
The source intentionally updates dqo/dko in place; their own role is exempt. -/
def Isolated (b : Buffers h kind) (p : Phase) (prog : Program h p) (out : Outputs b p prog) : Prop :=
  ∀ r i,active kind h p prog r i → ∀ q,q∈FPTile.inputs kind → q≠r →
    ∀ j,j<(b.tensor q).metadata.numel → Memory.Apart (out.toPlacement.address r i) ((b.view q).address j)
def InputFrame {β Store Byte : Type} (api : Memory.API β Store Byte) (b : Buffers h kind)
    (p : Phase) (prog : Program h p) (out : Outputs b p prog)
    (observe : RegionName → Nat → Cell α → β) (v : State α) (initial : Store) : Prop :=
  ∀ q,q∈FPTile.inputs kind → q∉FPTile.outputs kind → ∀ j,j<(b.tensor q).metadata.numel →
    api.read (Shared.commit api out.toPlacement observe (relocated h p v) initial) ((b.view q).address j)=
      api.read initial ((b.view q).address j)
theorem input_frame {β Store Byte : Type} (api : Memory.API β Store Byte) (b : Buffers h kind)
    (p : Phase) (prog : Program h p) (out : Outputs b p prog) (isolated : Isolated b p prog out)
    (observe : RegionName → Nat → Cell α → β) (v : State α) (initial : Store) :
    InputFrame api b p prog out observe v initial := by
  intro q hq hnot j hj
  apply Memory.run_read_frame
  intro e he
  obtain ⟨a,ha,rfl⟩ := List.mem_map.mp he
  have active := (out.coverage a.1 a.2).mp ha
  exact isolated a.1 a.2 active q hq (by intro eq; subst q; exact hnot active.1) j hj

theorem real_output {Store Byte : Type} (api : Memory.API ℝ Store Byte)
    (b : Buffers h kind) (p : Phase) (prog : Program h p) (out : Outputs b p prog)
    (v : State ℝ) (initial : Store) (r : RegionName) (hr : r∈FPTile.outputs kind)
    (t : Fin p.span) (j : Fin h.BK) :
    api.read (Shared.commit api out.toPlacement (fun _ _ cell => cell.read .real) (relocated h p v) initial)
      ((b.tensor r).storage.address ((b.tensor r).physicalIndex (Geometry.offset h p prog t.val j.val)))=
      (v.mem r (FPPrefix.off h p prog t.val j)).read .real := by
  rw [←out.same r hr]
  rw [VeriTile.Bench.Repairs.FastLayernorm.KernelStorage.commit_output api out.toPlacement
    (fun _ _ cell => cell.read .real) (relocated h p v) initial r _ ⟨hr,t,j,rfl⟩]
  rw [relocated_output]

theorem native_output {Store Byte : Type} (api : Memory.API Native.Word Store Byte)
    (b : Buffers h kind) (p : Phase) (prog : Program h p) (out : Outputs b p prog)
    (v : State Native.Node) (initial : Store) (r : RegionName) (hr : r∈FPTile.outputs kind)
    (t : Fin p.span) (j : Fin h.BK) :
    api.read (Shared.commit api out.toPlacement (fun _ _ => KernelNative.observe) (relocated h p v) initial)
      ((b.tensor r).storage.address ((b.tensor r).physicalIndex (Geometry.offset h p prog t.val j.val)))=
      KernelNative.observe (v.mem r (FPPrefix.off h p prog t.val j)) := by
  rw [←out.same r hr]
  rw [VeriTile.Bench.Repairs.FastLayernorm.KernelStorage.commit_output api out.toPlacement
    (fun _ _ => KernelNative.observe) (relocated h p v) initial r _ ⟨hr,t,j,rfl⟩]
  rw [relocated_output]

#axiomsClean physical_eq_logical
#axiomsClean coverage
#axiomsClean native_input
#axiomsClean native_output
#axiomsClean input_frame
end
end VeriTile.Bench.Repairs.DecayCumsum.KernelTensor
