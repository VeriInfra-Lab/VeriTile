import bench.repairs.tritonbench_g.bmm_chunk_bwd.Proofs.FP
import bench.repairs.tritonbench_g.bmm_chunk_bwd.Proofs.Real
import VeriTile.Meta.Specification

namespace VeriTile.Bench.Repairs.BmmChunkBwd.HistoricalFPEquiv
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.BmmChunkBwd
open DirectFP NativeGraph
noncomputable section
open Classical
set_option maxHeartbeats 6000000

/-- Complete physical result and source certificates for the actual reached
trial/chosen states. All numerical outputs remain concrete graph observations. -/
specification bmm_chunk_bwd_native (rest : Algebra Expr) (actual : Arithmetic)
    (p : Metadata) (h : InputContract p) (a : Direct.Allocation p) (heap : Heap)
    (cs : List (Fin 9)) (chosen : Fin 9) :
    ∃ out : DirectFP.Result,runWrapper rest actual p h a cs chosen heap=.ok out ∧
      out.object=a.object ∧ out.pointer=a.pointer ∧ out.shape=(p.tensor .a).shape ∧
      out.device=(p.tensor .a).device ∧ out.dtype=.floating .fp32 ∧
      (p.hasOutput=Bool.true → out.object=a.objects .output ∧ out.pointer=base p .output) ∧
      (p.hasOutput=Bool.false → ∀ r,Used p r → out.object≠a.objects r) ∧
      Good (finalState rest p h a cs chosen heap) ∧
      (0<(p.tensor .a).numel →
        Certified rest p h a cs (initial p a heap) ∧
        SourceLaunch rest p h a chosen (trials rest p h a cs (initial p a heap))) ∧
      (∀ off,off<(p.tensor .a).numel → ∀ b : Fin 4,
        out.heap (a.pointer+4*off+b.val)=
          byte (interpret actual (graph p (finalState rest p h a cs chosen heap) off)) b.val) ∧
      (∀ address,¬(a.pointer≤address ∧ address<a.pointer+4*(p.tensor .a).numel) → out.heap address=heap address) := by
  let final := finalState rest p h a cs chosen heap
  refine ⟨result p a actual final heap,?_,rfl,rfl,rfl,rfl,rfl,?_,?_,?_,?_,?_,?_⟩
  · simp [runWrapper,(checkInputs_iff p).mpr h,final]
  · intro supplied; simp [result,Direct.Allocation.object,Direct.Allocation.pointer,supplied]
  · intro absent r used
    simpa [result,Direct.Allocation.object,absent] using a.freshIdentity absent r used
  · exact execute_good rest p h a cs chosen _ (initial_good p a heap)
  · intro positive
    exact ⟨certified rest p h a positive cs _ (initial_good p a heap),
      source_one rest p h a positive chosen _ (trials_good rest p h a cs _ (initial_good p a heap))⟩
  · exact output_byte p a actual final heap
  · exact output_frame p a actual final heap

/-- Every logical tensor coordinate has an actual chosen source writer and
its complete independent blocked FP recurrence, hence four defined bytes. -/
theorem native_outputs (rest : Algebra Expr) (actual : Arithmetic) (p : Metadata) (h : InputContract p)
    (a : Direct.Allocation p) (heap : Heap) (cs : List (Fin 9)) (chosen : Fin 9)
    (b s g n : Nat) (hb : b<batch p) (hs : s<seqlen p) (hg : g<groups p) (hn : n<width p) :
    let after := trials rest p h a cs (initial p a heap)
    ∃ q∈HostFP.grid (call p) (tuning chosen),
      ∃ i : TileIndex [(tuning chosen).BM,(tuning chosen).BN],
        FPReference.active (HostFP.launch (call p) (tuning chosen) q after) (tuning chosen).BM (tuning chosen).BN
          (HostContract.parameters (call p)) i ∧
        FPFlat.outOffset (call p) (tuning chosen) q after i=Direct.tensorIndex p b s g n ∧
        ∀ byteIndex : Fin 4,
          (result p a actual (finalState rest p h a cs chosen heap) heap).heap
            (a.pointer+4*Direct.tensorIndex p b s g n+byteIndex.val)=
          byte (interpret actual (FPReference.expected (algebra rest) (dot chosen) (tuning chosen).BM (tuning chosen).BN
            (tuning chosen).BCS (HostContract.parameters (call p))
            (HostFP.launch (call p) (tuning chosen) q (FPFlat.lift (call p) (Direct.placement p) after)) i)) byteIndex.val := by
  dsimp only
  let after := trials rest p h a cs (initial p a heap)
  have bound := Direct.tensor_bound p h b s g n hb hs hg hn
  have positive : 0<(p.tensor .a).numel := Nat.zero_lt_of_lt bound
  obtain ⟨q,hq,i,hi,address,_⟩ := Direct.coordinate_covered p h ⟨fun _ => 0,fun _ => 0,fun _ => 0,fun _ => 0⟩ chosen b s g n hb hs hg hn
  rw [←Direct.grid_eq p h positive] at hq
  rw [←Direct.parameters_eq p h positive] at hi address
  let q' : HostFP.Pid := (q.tile,q.batch,q.chunkHead)
  have scheduled : q'∈HostFP.grid (call p) (tuning chosen) := by
    have bounds := (HostGrid.mem_programs _ _).mp hq
    simpa [q',HostFP.grid,HostContract.grid,HostGrid.dimensions,HostContract.parameters] using bounds
  have active : FPReference.active (HostFP.launch (call p) (tuning chosen) q' after) (tuning chosen).BM (tuning chosen).BN
      (HostContract.parameters (call p)) i := hi
  have offset : FPFlat.outOffset (call p) (tuning chosen) q' after i=Direct.tensorIndex p b s g n := address
  refine ⟨q',scheduled,i,active,offset,?_⟩
  intro j
  change outputHeap p a actual (finalState rest p h a cs chosen heap) heap _=_
  rw [output_byte p a actual _ heap _ bound]
  have read := one_readback rest p h a positive chosen after
    (trials_good rest p h a cs _ (initial_good p a heap)) q' scheduled i active
  rw [offset] at read
  have final : finalState rest p h a cs chosen heap=one rest p h a chosen after := by
    simp [finalState,execute,show (p.tensor .a).numel≠0 by omega,after]
  unfold graph
  rw [final,read]
  rfl

/-- Only scalar primitive and fixed-K8 coordinate agreements, recursively at
actual operands and reached carries. No structural law or whole dot result. -/
def TaskNumerics (rest : Algebra Expr) (actual reference : Arithmetic) (p : Metadata)
    (h : InputContract p) (a : Direct.Allocation p) (heap : Heap) (cs : List (Fin 9)) (chosen : Fin 9) : Prop :=
  ∀ off,off<(p.tensor .a).numel → Numerics actual reference (graph p (finalState rest p h a cs chosen heap) off)

specification bmm_chunk_bwd_if_local (rest : Algebra Expr) (actual reference : Arithmetic)
    (p : Metadata) (h : InputContract p) (a : Direct.Allocation p) (heap : Heap)
    (cs : List (Fin 9)) (chosen : Fin 9) (laws : TaskNumerics rest actual reference p h a heap cs chosen) :
    ∀ off,off<(p.tensor .a).numel → ∀ b : Fin 4,
      (result p a actual (finalState rest p h a cs chosen heap) heap).heap (a.pointer+4*off+b.val)=
        byte (interpret reference (graph p (finalState rest p h a cs chosen heap) off)) b.val := by
  intro off inside b
  rw [show (result p a actual (finalState rest p h a cs chosen heap) heap).heap _=_ from
    output_byte p a actual _ heap off inside b]
  rw [numerics_eq actual reference _ (laws off inside)]

#axiomsClean bmm_chunk_bwd_native
#axiomsClean native_outputs
#axiomsClean bmm_chunk_bwd_if_local
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.BmmChunkBwd.HistoricalFPEquiv
