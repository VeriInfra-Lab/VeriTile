import bench.repairs.tritonbench_g.chunk_retention.Proofs.KernelTensorLayout

namespace VeriTile.Bench.Repairs.ChunkRetention.Whole001.KernelTensor
open VeriTile Triton FP.Structural FP.ProfiledRewrite KernelModel KernelPacket KernelStaging
noncomputable section
variable {α β : Type} [Inhabited α]
set_option maxHeartbeats 5000000

/-- The region input is constructed from the one caller backing read at the
actual Tensor data pointer, including the observed storage offset. -/
def initial (views : Binding c) (read : Nat→α) (pid grid : Nat→Nat) : State α where
  mem := fun r off=>.mk .real (read (address (views r) off))
  regs := fun _ _ _=>none
  pids := pid
  numPids := grid
  undef := fun d _ _=>defaultValue d

theorem input_read (views : Binding c) (read : Nat→α) (pid grid : Nat→Nat)
    (r : RegionName) (off : Nat) (hi : off<(views r).tensor.metadata.numel) :
    ((initial views read pid grid).mem r off).read .real=logical read (views r) off := by
  change read (address (views r) off)=_
  rw [logical,coordinates (views r) off hi]

theorem input_matrix_read (views : Binding c) (read : Nat→α) (pid grid : Nat→Nat)
    (r : RegionName) (rows cols : Nat) (hs : regionShape c r=[c.B,c.H,rows,cols])
    (head row col : Nat) (hh : head<c.B*c.H) (hr : row<rows) (hc : col<cols) :
    ((initial views read pid grid).mem r (head*(rows*cols)+row*cols+col)).read .real=
      logical read (views r) (head*(rows*cols)+row*cols+col) :=
  input_read views read pid grid r _ (input_matrix_bound c (views r) rows cols hs head row col hh hr hc)

def ReadRegion (a : Call) (r : RegionName) : Prop := match a.stage with
  | .forwardH=>r="k" ∨ r="v" ∨ (a.seed=Bool.true ∧ r="h0")
  | .forwardO=>r="q" ∨ r="k" ∨ r="v" ∨ r="h"
  | .backwardH=>r="q" ∨ r="incoming" ∨ (a.seed=Bool.true ∧ r="terminal")
  | .backwardQKV=>r="q" ∨ r="k" ∨ r="v" ∨ r="incoming" ∨ r="h" ∨ r="dh"
def Apart (x y : VeriTile.Torch.FP32.Tensor) : Prop :=
  x.storage.base+4*x.storage.elements≤y.storage.base ∨
    y.storage.base+4*y.storage.elements≤x.storage.base
def Inside (x : VeriTile.Torch.FP32.Tensor) (addr : Nat) : Prop :=
  x.storage.base≤addr ∧ addr<x.storage.base+4*x.storage.elements
def OutputSeparated (a : Call) (s : State α) (views : Binding a.shape) : Prop :=
  ∀ o p,Active a s o → Active a s p → o.region≠p.region →
    Apart (views o.region).tensor (views p.region).tensor
def InputSeparated (a : Call) (s : State α) (views : Binding a.shape) : Prop :=
  ∀ o,Active a s o → ∀ r,ReadRegion a r → Apart (views o.region).tensor (views r).tensor

/-- The union of every active output store in this one selected program. -/
def Written (a : Call) (s : State α) (views : Binding a.shape) (addr : Nat) : Prop :=
  ∃ o,Active a s o ∧ addr=address (views o.region) (o.offset s)

/-- One shared post-backing read for all outputs. Separation makes any witness
of a physical store select the same region cell. All other words are retained. -/
def project (decode : α→β) (read : Nat→β) (a : Call) (s : State α)
    (views : Binding a.shape) (after : State α) (addr : Nat) : β := by
  classical
  exact if h : Written a s views addr then
    let o:=Classical.choose h
    decode ((after.mem o.region (o.offset s)).read .real)
  else read addr

theorem same_word (a : Call) (s : State α) (legal : Legal a s) (views : Binding a.shape)
    (apart : OutputSeparated a s views) (o p : Observation a.shape)
    (ho : Active a s o) (hp : Active a s p)
    (he : address (views o.region) (o.offset s)=address (views p.region) (p.offset s)) :
    o.region=p.region ∧ o.offset s=p.offset s := by
  classical
  by_cases hr : o.region=p.region
  · refine ⟨hr,?_⟩
    rw [hr] at he
    unfold address at he
    omega
  · have hsep:=apart o p ho hp hr
    have hlo:=address_inside (views o.region) _ (observation_bound a s legal o ho (views o.region))
    have hlp:=address_inside (views p.region) _ (observation_bound a s legal p hp (views p.region))
    rcases hsep with hsep|hsep <;> omega

theorem projected (decode : α→β) (read : Nat→β) (a : Call) (s : State α)
    (legal : Legal a s) (views : Binding a.shape) (apart : OutputSeparated a s views)
    (after : State α) (o : Observation a.shape) (ho : Active a s o) :
    logical (project decode read a s views after) (views o.region) (o.offset s)=
      decode ((after.mem o.region (o.offset s)).read .real) := by
  classical
  rw [logical,coordinates _ _ (observation_bound a s legal o ho (views o.region))]
  have hw : Written a s views (address (views o.region) (o.offset s)):=⟨o,ho,rfl⟩
  rw [project,dif_pos hw]
  have hc:=Classical.choose_spec hw
  obtain ⟨hr,hi⟩:=same_word a s legal views apart (Classical.choose hw) o hc.1 ho hc.2.symm
  simp only [hr,hi]

theorem exterior (decode : α→β) (read : Nat→β) (a : Call) (s : State α)
    (views : Binding a.shape) (after : State α) (addr : Nat) (h : ¬Written a s views addr) :
    project decode read a s views after addr=read addr := by
  simp [project,h]

theorem input_frame (decode : α→β) (read : Nat→β) (a : Call) (s : State α)
    (legal : Legal a s) (views : Binding a.shape) (apart : InputSeparated a s views)
    (after : State α) (r : RegionName) (hr : ReadRegion a r) (addr : Nat)
    (inside : Inside (views r).tensor addr) : project decode read a s views after addr=read addr := by
  apply exterior
  rintro ⟨o,ho,he⟩
  have hb:=address_inside (views o.region) _ (observation_bound a s legal o ho (views o.region))
  have hs:=apart o ho r hr
  dsimp [Inside] at inside
  rcases hs with hs|hs <;> omega

#axiomsClean input_read
#axiomsClean input_matrix_read
#axiomsClean projected
#axiomsClean input_frame
#specNonCircular initial avoiding [run,FP.Structural.exec,KernelPacket.realTarget,KernelPacket.nativeTarget,project]
#specNonCircular OutputSeparated avoiding [run,FP.Structural.exec,KernelPacket.realTarget,KernelPacket.nativeTarget,project]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkRetention.Whole001.KernelTensor
