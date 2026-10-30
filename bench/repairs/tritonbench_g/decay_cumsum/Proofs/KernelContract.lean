import bench.repairs.tritonbench_g.decay_cumsum.Contract
import bench.repairs.tritonbench_g.decay_cumsum.Proofs.NativeABI

namespace VeriTile.Bench.Repairs.DecayCumsum.KernelContract
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

def pid (contract : InputContract h kind observed scaleIsFloat)
    (nonempty : 0<h.B*h.H*h.T*h.DK) (p : Phase) (hp : p∈phases h)
    (prog : Geometry.Program h p) : NativeIndex.PID h p := by
  have grid := contract.grid nonempty
  have count : p.count≤65535 := by
    simp only [phases,List.mem_append] at hp
    rcases hp with hp|hp
    · split_ifs at hp with hz
      · simp at hp
      · simp only [List.mem_singleton] at hp; subst p; exact grid.2.1
    · split_ifs at hp with hz
      · simp at hp
      · simp only [List.mem_singleton] at hp; subst p; norm_num [tailPhase]
  exact ⟨prog,prog.featureInside.trans grid.1,by have a:=prog.chunkInside;omega,
    by have a:=prog.rowInside; change prog.row<h.B*h.H at a;omega⟩

#axiomsClean checkInput_iff
end VeriTile.Bench.Repairs.DecayCumsum.KernelContract
