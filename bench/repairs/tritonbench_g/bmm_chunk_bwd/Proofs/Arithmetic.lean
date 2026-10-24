import bench.repairs.tritonbench_g.bmm_chunk_bwd.Proofs.NativeGraph
import bench.repairs.tritonbench_g.bmm_chunk_fwd.Proofs.K8Plan

namespace VeriTile.Bench.Repairs.BmmChunkBwd.DirectFP
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.BmmChunkBwd
open NativeGraph
noncomputable section
open Classical

abbrev Word := BitVec 32

def config (choice : Fin 9) : VeriTile.Bench.TritonBenchG.BmmChunkFwd.Host.Config :=
  ⟨(tuning choice).BM,(tuning choice).BN,(tuning choice).BCS,(tuning choice).stages,(tuning choice).warps⟩

theorem config_legal (choice : Fin 9) : config choice∈VeriTile.Bench.TritonBenchG.BmmChunkFwd.Host.configurations := by
  fin_cases choice <;> decide

/-- Only numerical machine operations are ports. Raw storage leaves are
identity and have no deferred correspondence; pending standalone dots cannot
reach a materialized observation. -/
structure Arithmetic where
  literal : Option ComputeDType → FloatDType → ℝ → Word
  cast : Option ComputeDType → FloatDType → FloatDType → Word → Word
  binary : Option ComputeDType → FloatDType → Binary → Word → Word → Word
  mma8 : BmmChunkFwd.K8.Primitive

/-- Structural expansion of every variable-sized source dot into fixed K8
instructions. The graph's actual carry and its initialization/live selector
are threaded through all instructions in source reduction-coordinate order. -/
def interpret (a : Arithmetic) : Expr → Word
  | .leaf word => word
  | .literal p d v => a.literal p d v
  | .cast p l r x => a.cast p l r (interpret a x)
  | .binary p d op x y => a.binary p d op (interpret a x) (interpret a y)
  | .pendingDot .. => 0
  | .dotAcc c p l r out opts batch m k n left right acc i =>
      BmmChunkFwd.K8.plan a.mma8 (config c) batch m k n (fun j => interpret a (left j))
        (fun j => interpret a (right j)) (interpret a acc) Bool.true i

/-- All source numerical correspondences are per primitive at this node's
actual operands. No successful call, address law, reduction result or complete
execution equality is a premise. A consumed dot is one configured dotAcc with
its original FP32 carry and explicit TF32 K8 instruction schedule, never a whole-dot equality. -/
def Numerics (actual reference : Arithmetic) : Expr → Prop
  | .leaf _ => True
  | .literal p d v => actual.literal p d v=reference.literal p d v
  | .cast p l r x => Numerics actual reference x ∧
      actual.cast p l r (interpret actual x)=reference.cast p l r (interpret actual x)
  | .binary p d op x y => Numerics actual reference x ∧ Numerics actual reference y ∧
      actual.binary p d op (interpret actual x) (interpret actual y)=reference.binary p d op (interpret actual x) (interpret actual y)
  | .pendingDot .. => False
  | .dotAcc c p l r out opts batch m k n a b acc i =>
      (∀ j,Numerics actual reference (a j)) ∧ (∀ j,Numerics actual reference (b j)) ∧
      Numerics actual reference acc ∧
      BmmChunkFwd.K8.Agreements actual.mma8 reference.mma8 (BmmChunkFwd.K8.instruction (config c)) (BmmChunkFwd.K8.coordinate (config c) batch i)
        (BmmChunkFwd.K8.left batch m k n (fun j => interpret actual (a j)) i)
        (BmmChunkFwd.K8.right batch m k n (fun j => interpret actual (b j)) i)
        (interpret actual acc) Bool.true ((k+7)/8)


theorem numerics_eq (actual reference : Arithmetic) (x : Expr) (h : Numerics actual reference x) :
    interpret actual x=interpret reference x := by
  induction x with
  | leaf w => rfl
  | literal p d v => exact h
  | cast p l r x ih =>
    change actual.cast p l r (interpret actual x)=_
    rw [h.2,ih h.1]; rfl
  | binary p d op x y ihx ihy =>
    change actual.binary p d op (interpret actual x) (interpret actual y)=_
    rw [h.2.2,ihx h.1,ihy h.2.1]; rfl
  | pendingDot => exact h.elim
  | dotAcc c p l r out opts batch m k n a b acc i iha ihb ihacc =>
    change BmmChunkFwd.K8.plan actual.mma8 (config c) batch m k n _ _ _ _ i=_
    unfold BmmChunkFwd.K8.plan
    rw [BmmChunkFwd.K8.transport actual.mma8 reference.mma8 _ _ _ _ _ _ _ h.2.2.2]
    have ha : (fun j => interpret actual (a j))=(fun j => interpret reference (a j)) := funext (fun j => iha j (h.1 j))
    have hb : (fun j => interpret actual (b j))=(fun j => interpret reference (b j)) := funext (fun j => ihb j (h.2.1 j))
    rw [ha,hb,ihacc h.2.2.1]; rfl


#axiomsClean config_legal
#axiomsClean numerics_eq
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.BmmChunkBwd.DirectFP
