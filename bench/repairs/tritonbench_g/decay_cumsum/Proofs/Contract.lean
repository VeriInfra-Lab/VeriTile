import bench.repairs.tritonbench_g.decay_cumsum.Contract
import bench.repairs.tritonbench_g.decay_cumsum.Proofs.WholeHost
import bench.repairs.tritonbench_g.decay_cumsum.Proofs.NativeABI

namespace VeriTile.Bench.Repairs.DecayCumsum
open VeriTile.Triton VeriTile.Torch

theorem tensorCheck_iff (h : Call) (device : Device) (x : TensorMetadata) :
    tensorCheck h device x=Bool.true ↔ TensorContract h device x := by
  have fields : TensorContract h device x ↔ x.layout=.strided ∧ x.dtype=.floating .fp32 ∧
      x.shape=[h.B,h.H,h.T,h.DK] ∧ x.device.kind="cuda" ∧ x.device=device ∧
      x.isContiguous=Bool.true ∧ x.isNeg=Bool.false ∧ (0<x.numel → x.dataPtr.toNat%4=0) := by
    constructor
    · intro a; exact ⟨a.layout,a.dtype,a.shape,a.cuda,a.device,a.contiguous,a.noNeg,a.alignment⟩
    · rintro ⟨a,b,c,d,e,f,g,i⟩; exact ⟨a,b,c,d,e,f,g,i⟩
  rw [fields]
  simp [tensorCheck,TensorMetadata.isCuda,Nat.pos_iff_ne_zero]
  tauto

theorem checkInput_iff (h : Call) (kind : Kind) (observed : RegionName → TensorMetadata)
    (scaleIsFloat : Bool) : checkInput h kind observed scaleIsFloat=.ok () ↔ InputContract h kind observed scaleIsFloat := by
  have fields : InputContract h kind observed scaleIsFloat ↔
      h.BK &&& (h.BK-1)=0 ∧ h.BK≤1048576 ∧ scaleIsFloat=Bool.true ∧
      (∀ r∈argumentRoles kind, TensorContract h (observed (firstRole kind)).device (observed r)) ∧
      (0<h.B*h.H*h.T*h.DK → h.features<2^31 ∧ h.T/h.BT≤65535 ∧ h.B*h.H≤65535) := by
    constructor
    · intro a; exact ⟨a.blockPower,a.blockLimit,a.scaleType,a.tensors,a.grid⟩
    · rintro ⟨a,b,c,d,e⟩; exact ⟨a,b,c,d,e⟩
  rw [fields]
  simp [checkInput,ite_eq_iff,List.all_eq_true,tensorCheck_iff]
  tauto

/-- Bind each modeled target to its original Tensor identity, metadata and
canonical active coordinates. No output payload is part of the binding. -/
structure TensorBinding (h : Call) (kind : Kind) (observed : RegionName → TensorMetadata)
    (targets : RegionName → Host.Target) : Prop where
  shape : ∀ r∈argumentRoles kind, (targets r).shape=(observed r).shape
  negative : ∀ r∈argumentRoles kind, (targets r).negative=(observed r).isNeg
  pointer : ∀ r∈argumentRoles kind, ∀ i, i<h.B*h.H*h.T*h.DK →
    (targets r).address i=(observed r).dataPtr.toNat+4*i

namespace InputContract
theorem arange (contract : InputContract h kind observed scaleIsFloat) : NativeABI.CompiledArange h :=
  ⟨contract.blockLimit⟩

def pid (contract : InputContract h kind observed scaleIsFloat)
    (nonempty : 0<h.B*h.H*h.T*h.DK) (p : Phase) (hp : p∈phases h)
    (prog : Geometry.Program h p) : NativeIndex.PID h p := by
  have grid := contract.grid nonempty
  have count : p.count≤65535 := by
    simp only [phases,List.mem_append] at hp
    rcases hp with hp|hp
    · split_ifs at hp with hz
      · simp at hp
      · simp only [List.mem_singleton] at hp
        subst p
        exact grid.2.1
    · split_ifs at hp with hz
      · simp at hp
      · simp only [List.mem_singleton] at hp
        subst p
        norm_num [tailPhase]
  exact ⟨prog,prog.featureInside.trans grid.1,by have a:=prog.chunkInside;omega,
    by have a:=prog.rowInside; change prog.row<h.B*h.H at a;omega⟩
end InputContract

namespace Host
noncomputable def guarded {β : Type} (h : Call) (kind : Kind)
    (observed : RegionName → TensorMetadata) (scaleIsFloat : Bool)
    (body : Except (Failure β) (Returned β)) (before : Copyback.Heap β) : Except (Failure β) (Returned β) :=
  match checkInput h kind observed scaleIsFloat with
  | .error _ => .error ⟨.validation,"unsupported input",before⟩
  | .ok () => body

theorem guarded_eq {β : Type} (contract : InputContract h kind observed scaleIsFloat)
    (body : Except (Failure β) (Returned β)) (before : Copyback.Heap β) :
    guarded h kind observed scaleIsFloat body before=body := by
  simp [guarded,(checkInput_iff h kind observed scaleIsFloat).mpr contract]
end Host
#axiomsClean checkInput_iff
#axiomsClean InputContract.arange
end VeriTile.Bench.Repairs.DecayCumsum
