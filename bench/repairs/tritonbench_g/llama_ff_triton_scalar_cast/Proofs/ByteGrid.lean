import bench.repairs.tritonbench_g.llama_ff_triton_scalar_cast.Proofs.Grid
import bench.repairs.tritonbench_g.llama_ff_triton_scalar_cast.Proofs.Words
import bench.repairs.tritonbench_g.llama_ff_triton_scalar_cast.Proofs.NativeReference

namespace VeriTile.Bench.Repairs.LlamaFFScalarCast.ByteGrid
open VeriTile.Triton FP.Structural Execution Words
noncomputable section
set_option maxHeartbeats 6000000

def target (a : Instructions) (p : Parameters) (mem : Memory) (x : Grid.Index p) : H :=
  half (NativeReference.target (arithmetic a) (gates a) p (entry p mem) x.1.val x.2.val)
def run (a : Instructions) (p : Parameters) (mem : Memory) :=
  Grid.grid Bool.true (arithmetic a) (gates a) p (entry p mem)
def Outside (p : Parameters) (r : RegionName) (b : Nat) :=
  r≠("OUT" : RegionName) ∨ ∀ x : Grid.Index p,b/2≠Grid.address p x
def Post (a : Instructions) (p : Parameters) (initial final : Memory) : Prop :=
  (∀ x : Grid.Index p,∀ b : Fin 2,final "OUT" (2*Grid.address p x+b.val)=encode16 (target a p initial x) b) ∧
  (∀ x : Grid.Index p,read16 final "OUT" (Grid.address p x)=target a p initial x) ∧
  (∀ r b,Outside p r b → final r b=initial r b)

theorem complete (a : Instructions) (p : Parameters) (mem : Memory) :
    ∃ final,run a p mem=some final ∧ Post a p mem (commit mem final) := by
  obtain ⟨final,he,values,frame⟩ := Grid.whole Bool.true (arithmetic a) (gates a) p (entry p mem) (entry_bytes p mem)
  have hv (x : Grid.Index p) (b : Fin 2) :
      commit mem final "OUT" (2*Grid.address p x+b.val)=encode16 (target a p mem x) b := by
    have hd : (2*Grid.address p x+b.val)/2=Grid.address p x := by omega
    have hm : (2*Grid.address p x+b.val)%2=b.val := by omega
    simp only [commit,ite_true,hd,hm,values x trivial,Cell.read_same,Grid.target,ite_true,
      NativeReference.logical_eq,target]
  refine ⟨final,he,hv,?_,?_⟩
  · intro x
    unfold read16
    rw [show commit mem final "OUT" (2*Grid.address p x)=encode16 (target a p mem x) 0 from by
      simpa only [Fin.val_zero,Nat.add_zero] using hv x 0]
    rw [hv x ⟨1,by decide⟩]
    exact decode_encode _
  · intro r b outside
    by_cases hr : r=("OUT" : RegionName)
    · subst r
      have ho : ∀ x : Grid.Index p,b/2≠Grid.address p x := by
        rcases outside with h|h
        · exact False.elim (h rfl)
        · exact h
      have hf := frame "OUT" (b/2) (Or.inr (fun x _ => ho x))
      have hi := congrFun (congrFun (commit_entry p mem) "OUT") b
      simp only [commit,ite_true] at hi ⊢
      rw [hf]
      exact hi
    · simp [commit,hr]

#axiomsClean complete
#specNonCircular target avoiding [run,Grid.grid,Grid.one,nativeExecution,logicalNative,rowAcc,rowMoment]
#specNonCircular Post avoiding [run,Grid.grid,Grid.one,nativeExecution]
end
end VeriTile.Bench.Repairs.LlamaFFScalarCast.ByteGrid
