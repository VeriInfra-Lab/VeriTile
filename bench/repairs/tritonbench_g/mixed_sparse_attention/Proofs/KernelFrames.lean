import bench.repairs.tritonbench_g.mixed_sparse_attention.Proofs.KernelReal
import bench.repairs.tritonbench_g.mixed_sparse_attention.Proofs.KernelNative

namespace VeriTile.Bench.Repairs.MixedSparseAttention.KernelFrames
open VeriTile Triton FP.Structural
open Whole002 Source Whole Checked KernelGrid KernelTensor
open VeriTile.Bench.Repairs.LayernormFwdTriton
noncomputable section
set_option maxHeartbeats 4000000

def commitSource {α β Store Byte : Type} [Inhabited α] (api : Memory.API β Store Byte)
    (c : Config) (request : Request) (before : State α)
    (observe : RegionName → Nat → Cell α → β) (final : State α) (store : Store) : Store :=
  Memory.run api ((cells c before).map fun a=>
    ⟨(request.tensor .output).pointer+4*a.2,observe a.1 a.2 (final.mem a.1 a.2)⟩) store

theorem commitSource_eq {α β Store Byte : Type} [Inhabited α] (api : Memory.API β Store Byte)
    (b : Buffers request) (accepted : Admission c request values) (before final : State α)
    (observe : RegionName → Nat → Cell α → β) (store : Store) :
    commitSource api c request before observe final store=
      Shared.commit api (placement b accepted before) observe final store := by
  change Memory.run api ((cells c before).map _) store=Memory.run api ((cells c before).map _) store
  congr 1
  apply List.map_congr_left
  intro a ha
  rw [placement_address b accepted before a.1 a.2 ((cells_coverage c before a.1 a.2).mp ha)]

theorem cells_empty {α : Type} [Inhabited α] (c : Config) (s : State α) (empty : c.Z=0 ∨ c.T=0) :
    cells c s=[] := by
  apply List.eq_nil_iff_forall_not_mem.mpr
  intro a ha
  obtain ⟨_,q,_,_⟩:=(cells_coverage c s a.1 a.2).mp ha
  rcases empty with h|h
  · have bound:=q.1.isLt
    simpa [h] using bound
  · have bound:=q.2.1.isLt
    simpa [h] using bound

theorem input_bytes {α β Store Byte : Type} [Inhabited α]
    (api : Memory.API β Store Byte) (b : Buffers request) (accepted : Admission c request values)
    (s final : State α) (observe : RegionName → Nat → Cell α → β) (store : Store)
    (r : Role) (hr : r∈inputRoles) (i : Nat) (hi : i<r.size c) (j : Fin 4) :
    api.byte (Shared.commit api (placement b accepted s) observe final store)
      ((b.tensor r).storage.address ((b.tensor r).physicalIndex i)+j.val)=
    api.byte store ((b.tensor r).storage.address ((b.tensor r).physicalIndex i)+j.val) := by
  rw [logical_address b accepted r i hi]
  apply Shared.commit_frame
  intro region offset hw
  rw [placement_address b accepted s region offset hw]
  obtain ⟨_,q,_,eq⟩:=hw
  have ho:=Raw.offset_bound q
  have sep:=accepted.fresh r hr
  have storage:=accepted.storage r
  have lane:=j.isLt
  rw [eq]
  omega

theorem inactive_bytes {α β Store Byte : Type} [Inhabited α]
    (api : Memory.API β Store Byte) (b : Buffers request) (accepted : Admission c request values)
    (s final : State α) (observe : RegionName → Nat → Cell α → β) (store : Store)
    (q : Raw.Coord c) (hq : ¬live c s q) (j : Fin 4) :
    api.byte (Shared.commit api (placement b accepted s) observe final store)
      ((b.tensor .output).storage.address ((b.tensor .output).physicalIndex (Raw.offset q))+j.val)=
    api.byte store ((b.tensor .output).storage.address ((b.tensor .output).physicalIndex (Raw.offset q))+j.val) := by
  rw [logical_address b accepted .output (Raw.offset q) (Raw.offset_bound q)]
  apply Shared.commit_frame
  intro r i hw
  rw [placement_address b accepted s r i hw]
  have ne : i≠Raw.offset q := by
    intro eq
    have region:=hw.1
    rw [region,eq] at hw
    exact inactive c s q hq hw
  have lane:=j.isLt
  omega

theorem empty {α Store Byte : Type} [Inhabited α]
    (M : Program → Algebra α) (D : Program → Dot α) (c : Config) (request : Request)
    (accepted : EmptyAdmission c request) (s : State α) (work : List Program)
    (order : work.Perm (programs (grid c))) (api : Memory.API β Store Byte)
    (observe : RegionName → Nat → Cell α → β) (store : Store) :
    runPrograms M D c (grid c) work s=some s ∧
      commitSource api c request s observe s store=store ∧
      (∀ byte,api.byte (commitSource api c request s observe s store) byte=api.byte store byte) := by
  have hbm : 0<c.BM := by rw [accepted.schema.selected.2.1];decide
  have hw : work=[] := by
    rw [no_programs c hbm accepted.empty] at order
    exact List.perm_nil.mp order
  simp [hw,runPrograms,commitSource,cells_empty c s accepted.empty,Memory.run]

#axiomsClean input_bytes
#axiomsClean inactive_bytes
#axiomsClean commitSource_eq
#axiomsClean empty
end
end VeriTile.Bench.Repairs.MixedSparseAttention.KernelFrames
