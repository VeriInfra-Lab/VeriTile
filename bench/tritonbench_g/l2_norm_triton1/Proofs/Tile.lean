import bench.tritonbench_g.l2_norm_triton1.Proofs.Scalar
import bench.tritonbench_g.swiglu_backward.Proofs.Mathematical_WholeTile

namespace VeriTile.Bench.Repairs.L2NormTriton1.Tile
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.SwigluBackward.Whole (writeCell scatter_mem fold_written fold_frame)
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000

/-- Epsilon is a genuine scalar ABI input, not represented as a finite Real
value in the FP specialization. All other registers start unspecified. -/
def rowState {α : Type} [Inhabited α] (row : Nat) (eps : α) (s : State α) : State α :=
  { s with pids := fun ax => if ax=0 then row else 0
           regs := (State.setReg s (dtype := .real) (shape := []) "eps" (fun _ => eps)).regs }

private theorem index_toNat (row N lane : Nat) :
    ((row : Int)*(N : Int)+(lane : Int)).toNat=row*N+lane := by
  rw [←Int.natCast_mul,←Int.natCast_add,Int.toNat_natCast]

private theorem index_lt (lane N : Nat) : ((lane : Int)<(N : Int)) ↔ lane<N := by omega

/-- Constructive execution of every statement of the complete typed source. -/
def sourceRun {α : Type} [Inhabited α] (M : Algebra α) (N B row : Nat)
    (eps : α) (s : State α) :
    {t : State α // run M (source N B).surfaceBody (rowState row eps s)=some t} := by
  refine ⟨?_,?_⟩
  swap
  simp (maxSteps := 900000) [source,template,ComputeKernel.surfaceBody,
    run,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,
    evalOffsets,evalDeltas,store,Option.bind,ComputeDType.eraseDType,
    numeric,numericLt,bop,natLt,rowState,ofFloat,toFloat,TileShape.axisDim,
    Region.cast,State.setReg,index_toNat,index_lt]
  rfl

def inputs {α : Type} [Inhabited α] (s : State α) (N row : Nat) : Nat → α :=
  fun i => (s.mem "X" (row*N+i)).read .real

def memory {α : Type} [Inhabited α] (M : Algebra α) (N B row : Nat)
    (eps : α) (s : State α) : RegionName → Nat → Cell α :=
  (TileShape.allIndices [B]).foldl (fun mem i =>
    if i.1.val<N then writeCell mem "Y" (row*N+i.1.val)
      (.mk .real (scalar M N B eps (inputs s N row) i.1.val))
    else mem) s.mem

theorem sourceRun_mem {α : Type} [Inhabited α] (M : Algebra α) (N B row : Nat)
    (eps : α) (s : State α) : (sourceRun M N B row eps s).val.mem=memory M N B row eps s := by
  simp only [sourceRun,store,scatter_mem,memory,scalar,variance,masked,loaded,inputs,FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,
    ←Int.natCast_mul,Int.toNat_natCast]
  congr 1
  funext mem i
  by_cases hi : i.1.val<N <;> simp only [hi,if_true,if_false]
  rfl


theorem readback {α : Type} [Inhabited α] (M : Algebra α) (N B row : Nat)
    (eps : α) (s : State α) (target : TileIndex [B]) (active : target.1.val<N) :
    (sourceRun M N B row eps s).val.mem "Y" (row*N+target.1.val)=
      .mk .real (scalar M N B eps (inputs s N row) target.1.val) := by
  rw [sourceRun_mem]
  have unique : ∀ i : TileIndex [B],i.1.val<N →
      row*N+i.1.val=row*N+target.1.val → i=target := by
    intro i _ h
    apply Prod.ext
    · apply Fin.ext; omega
    · exact Subsingleton.elim _ _
  obtain ⟨before,after,hall⟩ := List.append_of_mem (TileShape.mem_allIndices [B] target)
  unfold memory
  rw [hall,List.foldl_append,List.foldl_cons]
  simp only [active,if_true]
  apply fold_written _ _ _ _ target unique
  simp [writeCell]

theorem frame {α : Type} [Inhabited α] (M : Algebra α) (N B row : Nat)
    (eps : α) (s : State α) (region : RegionName) (address : Nat)
    (outside : region≠"Y" ∨ ∀ i : TileIndex [B],i.1.val<N → address≠row*N+i.1.val) :
    (sourceRun M N B row eps s).val.mem region address=s.mem region address := by
  rw [sourceRun_mem]
  exact fold_frame _ _ _ _ _ _ _ _ outside

#axiomsClean sourceRun
#axiomsClean readback
#axiomsClean frame
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.L2NormTriton1.Tile
