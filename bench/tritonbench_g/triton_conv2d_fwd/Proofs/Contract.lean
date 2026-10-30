import bench.tritonbench_g.triton_conv2d_fwd.Contract
import bench.tritonbench_g.triton_conv2d_fwd.Proofs.Safety

namespace VeriTile.Bench.Repairs.Conv2d
set_option maxHeartbeats 800000

theorem validCheck_iff (s : Shape) : validCheck s=Bool.true ↔ Valid s := by
  simp only [validCheck,decide_eq_true_eq]
  constructor
  · rintro ⟨a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,t,u,v⟩
    exact ⟨a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,t,u,v⟩
  · intro h
    exact ⟨h.cg,h.og,h.h,h.w,h.kh,h.kw,h.sh,h.sw,h.groups,h.oh,h.ow,h.shapeH,h.shapeW,h.strides⟩

theorem checkCall_iff (c : Call) : checkCall c=.ok () ↔ Domain c := by
  have domain : Domain c ↔ Valid c.shape ∧ TensorKind c.input ∧ TensorKind c.weight ∧
      MetadataBinding c ∧ c.input.tensor.device=c.weight.tensor.device ∧
      c.callerDevice=c.input.tensor.device.index ∧ Stored c.input (inputWords c.shape) ∧
      Stored c.weight (weightWords c.shape) ∧ envelope c.shape<2^31 ∧ LaunchGrid c.shape := by
    constructor
    · intro h;exact ⟨h.valid,h.inputKind,h.weightKind,h.binding,h.sameDevice,h.currentDevice,h.inputStorage,h.weightStorage,h.signed,h.grid⟩
    · rintro ⟨a,b,d,e,f,g,h,i,j,k⟩;exact ⟨a,b,d,e,f,g,h,i,j,k⟩
  rw [domain]
  simp only [checkCall]
  split_ifs <;> simp_all [validCheck_iff] <;> aesop

theorem check_iff (c : ClassifiedCall) : check c=.ok () ↔ Admitted c := by
  cases c <;> simp [check,Admitted,checkCall_iff]

theorem rejected_iff (c : ClassifiedCall) :
    (∃ e,check c=.error e) ↔ ¬Admitted c := by
  rw [←check_iff]
  cases check c <;> simp

theorem domain_safety (c : Call) (h : Domain c) :
    SourceSafe c.shape (inputWords c.shape) (weightWords c.shape) := by
  apply source_safe c.shape h.valid h.signed
  · intro hn;simp [inputWords,hn]
  · simp [weightWords]

/-- Bounds on the unchanged direct CUDA grid, derived from actual checked
metadata rather than a segmented launcher or an assumed successful grid. -/
theorem grid_limits (c : Call) (d : Domain c) (nonempty : 0<c.shape.n) :
    blocks c.shape.spatial 128<2^31 ∧ blocks c.shape.outputs 32≤65535 ∧ c.shape.groups≤65535 := by
  have bound:=parameters_bound c.shape (pSpatial c.shape) (by simp [scalarParameters])
  have signed:=d.signed
  unfold pSpatial at bound
  exact ⟨by omega,d.grid nonempty⟩

#axiomsClean validCheck_iff
#axiomsClean checkCall_iff
#axiomsClean check_iff
#axiomsClean rejected_iff
#axiomsClean domain_safety
#axiomsClean grid_limits
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.Conv2d
