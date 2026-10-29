import bench.repairs.tritonbench_g.token_attn_reduceV_grid_y_checked.Proofs.Interface
import bench.repairs.tritonbench_g.token_attn_reduceV_grid_y_checked.Proofs.Compatibility
import bench.repairs.tritonbench_g.token_attn_reduceV.Native

namespace VeriTile.Bench.Repairs.TokenAttnReduceVGridYChecked.Interface
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.TokenAttnReduceV
noncomputable section

def nativeIO (R : Native.Reductions) (p : Parameters) (B H : Nat)
    (counts : Fin B → Nat) (jobs : List Grid.PID) : ProgramIO Native.Term Native.Term :=
  programIO Native.algebra (Native.reduction R) p B H counts jobs

/-- Pure observation after source execution. The logical symbolic inputs stay
unrestricted, so old input leaves and mismatched-tag defaults are both retained. -/
def nativeObserver (I : Native.Instructions) (S : Native.Schedule) (p : Parameters) (B H : Nat)
    (q : Fin (outputs p B H).length) (_j : TileIndex (outputs p B H)[q].shape) : Native.Term → Native.Word :=
  Native.eval I S

def floating (I : Native.Instructions) (R : Native.Reductions) (S : Native.Schedule)
    (p : Parameters) (B H : Nat) (counts : Fin B → Nat)
    (v : Inputs Native.Term p B H counts) : Outputs Native.Word p B H :=
  result fun b h d => Native.eval I S (orderedRow Native.algebra (Native.reduction R) p
    (counts b) (probAt v b h) (valAt v b h) d)

theorem native_observes (I : Native.Instructions) (R : Native.Reductions) (S : Native.Schedule)
    (p : Parameters) (B H : Nat) (counts : Fin B → Nat) (jobs : List Grid.PID)
    (order : (Grid.fullGrid B H).Perm jobs) (separate : Grid.Separate p) (unique : Grid.Unique p jobs) :
    (nativeIO R p B H counts jobs).ObservesOn (nativeObserver I S p B H)
      (LengthsMatch p B H counts) (floating I R S p B H counts) := by
  have h := ProgramIO.ObservesOn.of_implementsOn
    (implements Native.algebra (Native.reduction R) p B H counts jobs order separate unique)
    (nativeObserver I S p B H)
  have eq : (fun v => (programIO Native.algebra (Native.reduction R) p B H counts jobs).mapOutputs (nativeObserver I S p B H)
      (orderedTarget Native.algebra (Native.reduction R) p B H counts v)) = floating I R S p B H counts := by
    funext v q j
    have hq : q.val<1 := q.isLt
    have eq : q=⟨0,by simp [nativeIO,programIO,memory,outputs]⟩ := Fin.ext (show q.val=0 by omega)
    subst q
    rfl
  exact Eq.mp (congrArg (fun f => (nativeIO R p B H counts jobs).ObservesOn
    (nativeObserver I S p B H) (LengthsMatch p B H counts) f) eq) h

/-- Exact preservation of the old arbitrary typed word heap and native
reference, including nonfloating tags and every metadata alias allowed there. -/
theorem old_native_formula (I : Native.Instructions) (R : Native.Reductions) (S : Native.Schedule)
    (p : Parameters) (B H : Nat) (m : Grid.Memory Native.Word)
    (b : Fin B) (h : Fin H) (d : Fin p.width) :
    floating I R S p B H (heapCounts p B (Native.entry m))
      (fromHeap p B H (heapCounts p B (Native.entry m)) (Native.entry m))
      ⟨0,by simp [outputs]⟩ (b,h,d,PUnit.unit) =
      Native.eval I S (Grid.values Native.algebra (Native.reduction R) p (Native.entry m)
        ((b.val,h.val),(d,PUnit.unit))) :=
  congrArg (Native.eval I S) (heap_formula Native.algebra (Native.reduction R) p B H (Native.entry m) b h d)

#axiomsClean native_observes
#axiomsClean old_native_formula
end
end VeriTile.Bench.Repairs.TokenAttnReduceVGridYChecked.Interface
