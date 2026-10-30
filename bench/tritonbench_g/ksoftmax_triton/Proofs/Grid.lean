import bench.tritonbench_g.ksoftmax_triton.Proofs.FP
import bench.tritonbench_g.ksoftmax_triton.Proofs.Native

namespace VeriTile.Bench.Repairs.Ksoftmax
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.KsoftmaxTriton
noncomputable section

inductive Operation where
  | forward (log causal : Bool) (mode : Forward.MaskMode)
  | backward (log causal : Bool)

def atProgram {α : Type} (s : State α) (batch row : Nat) : State α :=
  {s with pids := fun ax => if ax=0 then batch else if ax=1 then row else s.pids ax}

def runRow {α : Type} [Inhabited α] (M : Algebra α) (op : Operation)
    (Dst Left Right : RegionName) (Q K D : Nat) (hD : 0<D)
    (batch row : Nat) (s : State α) : State α :=
  let initial := atProgram s batch row
  match op with
  | .forward log causal mode =>
    (Floating.forward_run M Dst Left Right (Q*K) K (Q*K) K K K D hD log mode causal initial).val
  | .backward log causal =>
    (Floating.backward_run M Dst Left Right (Q*K) K (Q*K) K (Q*K) K K D log causal initial).val

/-- Every row is the actual typed source execution. The grid list retains
program ownership; it is not an assumed output array or callback. -/
def runRows {α : Type} [Inhabited α] (M : Algebra α) (op : Operation)
    (Dst Left Right : RegionName) (B Q K D : Nat) (hD : 0<D)
    (rows : List (Fin B × Fin Q)) (s : State α) : State α :=
  rows.foldl (fun t p => runRow M op Dst Left Right Q K D hD p.1 p.2 t) s

def programs (B Q : Nat) : List (Fin B × Fin Q) :=
  (List.finRange B).flatMap fun b => (List.finRange Q).map fun row => (b,row)

theorem row_frame {α : Type} [Inhabited α] (M : Algebra α) (op : Operation)
    (Dst Left Right : RegionName) (B Q K D : Nat) (hD : 0<D)
    (batch : Fin B) (row : Fin Q) (s : State α) (r : RegionName) (offset : Nat)
    (outside : r≠Dst ∨ B*Q*K≤offset) :
    (runRow M op Dst Left Right Q K D hD batch row s).mem r offset=s.mem r offset := by
  have miss : r≠Dst ∨ ∀ i : Fin D, i.val<K →
      offset≠(atProgram s batch row).pids 0*(Q*K)+(atProgram s batch row).pids 1*K+i.val := by
    rcases outside with h|h
    · exact Or.inl h
    · right
      intro i hi heq
      have bound:=Native.active_bound batch.isLt row.isLt hi
      simp only [atProgram,ite_true,show ¬(1:Nat)=0 by decide,ite_false] at heq
      unfold Native.flat at bound
      omega
  cases op with
  | forward log causal mode =>
    exact Floating.forward_frame M Dst Left Right (Q*K) K (Q*K) K K K D hD log mode causal
      (atProgram s batch row) r offset miss
  | backward log causal =>
    exact Floating.backward_frame M Dst Left Right (Q*K) K (Q*K) K (Q*K) K K D log causal
      (atProgram s batch row) r offset miss

theorem rows_frame {α : Type} [Inhabited α] (M : Algebra α) (op : Operation)
    (Dst Left Right : RegionName) (B Q K D : Nat) (hD : 0<D)
    (rows : List (Fin B × Fin Q)) (s : State α) (r : RegionName) (offset : Nat)
    (outside : r≠Dst ∨ B*Q*K≤offset) :
    (runRows M op Dst Left Right B Q K D hD rows s).mem r offset=s.mem r offset := by
  induction rows generalizing s with
  | nil => rfl
  | cons p rest ih =>
    exact (ih _).trans (row_frame M op Dst Left Right B Q K D hD p.1 p.2 s r offset outside)

#axiomsClean row_frame
#axiomsClean rows_frame
end
end VeriTile.Bench.Repairs.Ksoftmax
