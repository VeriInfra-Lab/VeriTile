import bench.repairs.tritonbench_g.attn_fwd_triton_tile_checked.Contract
import bench.repairs.tritonbench_g.attn_fwd_triton.Proofs.CheckedFPEquiv

namespace VeriTile.Bench.Repairs.AttnFwdTritonTileChecked.Address
open VeriTile Triton AttnFwdTriton Checked TypedExecution
open Verified.PhysicalCells (Role Layout width)
open VeriTile.Bench.Repairs.LayerNormFwd.Address (signed64 wrap64)

/-- The existing host envelope bounds every active source role offset, using
only source geometry. Padded/advanced matrix lanes are covered by the broader
Checked.matrix_formed theorem retained in the public contract import. -/
theorem active_bound (p : Config) (z h n d : Nat) (r : Role)
    (ha : roleActive p z h n d r) : roleOffset p z h n d r<envelope (typedGeometry p) := by
  have matrix (r : Role) (hqs : r≠Role.queryScale) (hks : r≠Role.keyScale)
      (active : z<p.batches ∧ h<p.heads ∧ n<p.context ∧ d<p.headDim) :
      ((typedGeometry p).stride r).at z h n d<envelope (typedGeometry p) :=
    matrix_formed _ r hqs hks z h n d (Nat.le_of_lt active.1) (Nat.le_of_lt active.2.1)
      (by have:=active.2.2.1; change n≤p.context+256; omega)
      (Nat.le_trans (Nat.le_of_lt active.2.2.2) (HostShape.blockD_covers _))
  cases r
  · exact matrix .query (by decide) (by decide) ha
  · exact matrix .key (by decide) (by decide) ha
  · exact matrix .value (by decide) (by decide) ha
  · exact matrix .output (by decide) (by decide) ha
  · have hbound:=ha.2.2.2
    simp only [roleOffset,envelope,typedGeometry,countQ,countK]
    omega
  · have hbound:=ha.2.2.2
    simp only [roleOffset,envelope,typedGeometry,countQ,countK]
    omega

theorem native_active (p : Config) (a : Layout) (base : Role→Nat) (views : Views a p base)
    (qs ks : TensorMetadata) (hl : launchChecks (typedGeometry p) qs ks=Bool.true)
    (hp : pointerChecks (typedGeometry p) base=Bool.true)
    (z h n d : Nat) (r : Role) (ha : roleActive p z h n d r) :
    machineOffset p z h n d r=roleOffset p z h n d r ∧
    signed64 (machineOffset p z h n d r)=(roleOffset p z h n d r : Int) ∧
    Verified.NativeOffsets.bytePointer (base r) (width r) (machineOffset p z h n d r)=
      a.address (roleKey p z h n d r) := by
  have hb:=active_bound p z h n d r ha
  have hs:=formed_signed (typedGeometry p) qs ks (inherited _ _ _ hl) _ hb
  have hp':=formed_pointer (typedGeometry p) base hp r _ hb
  have hm : machineOffset p z h n d r=roleOffset p z h n d r := by
    rw [machineOffset_mod,hs.2]
  exact ⟨hm,hm ▸ hs.1,by rw [hm,hp'.2.2,views.address z h n d r ha]⟩

theorem control (p : Config) (qs ks : TensorMetadata)
    (hl : launchChecks (typedGeometry p) qs ks=Bool.true) :
    ControlCertificate p (HostShape.blockD p.headDim) :=
  control_certificate p _ (Checked.launchable _ _ _ (inherited _ _ _ hl))

#axiomsClean active_bound
#axiomsClean native_active
#axiomsClean control
end VeriTile.Bench.Repairs.AttnFwdTritonTileChecked.Address
