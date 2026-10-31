import bench.tritonbench_g.fast_rms_layernorm.Proofs.FPReference
import VeriTile.Triton.Float.ProfiledRewrite

namespace VeriTile.Bench.Repairs.FastRmsLayernorm.FPSource
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 25000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def update (r : RegionName) (o : Nat) (v : Cell α) (heap : RegionName → Nat → Cell α) :=
  fun q n => if q=r ∧ n=o then v else heap q n

def commit {ι : Type} (r : RegionName) (off : ι → Nat) (val : ι → Cell α)
    (active : ι → Prop) [DecidablePred active] (indices : List ι) (heap : RegionName → Nat → Cell α) :=
  indices.foldl (fun mem i => if active i then update r (off i) (val i) mem else mem) heap

theorem fold_mem {ι : Type} (r : RegionName) (off : ι → Nat) (val : ι → Cell α)
    (active : ι → Prop) [DecidablePred active] (indices : List ι) (u : State α) :
    (indices.foldl (fun s i => if active i then s.write r (off i) (val i) else s) u).mem=
      commit r off val active indices u.mem := by
  induction indices generalizing u with
  | nil => rfl
  | cons i rest ih =>
    by_cases hi : active i <;> simp only [List.foldl_cons,hi,ite_true,ite_false]
    · rw [ih]
      simp [commit,List.foldl_cons,hi,State.write,update] <;> rfl
    · rw [ih]
      simp [commit,List.foldl_cons,hi,State.write,update] <;> rfl

def forwardHeap (M : Algebra α) (h : Call) (u : State α) : RegionName → Nat → Cell α :=
  let x:=fun j : Fin h.block => (u.mem "x" (u.pids 0*h.cols+j.val)).read .real
  let w:=fun j : Fin h.block => (u.mem "w" j.val).read .real
  commit "y" (fun i : TileIndex [h.block] => u.pids 0*h.cols+i.1.val)
    (fun i => FPReference.output M h x w i.1) (fun i => i.1.val<h.cols) (TileShape.allIndices [h.block])
    (update "r" (u.pids 0) (FPReference.savedInv M h x) u.mem)

end
end VeriTile.Bench.Repairs.FastRmsLayernorm.FPSource
