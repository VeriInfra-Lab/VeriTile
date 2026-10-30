import bench.tritonbench_g.bmm_optimized.Proofs.IOComplete

namespace VeriTile.Bench.Repairs.BmmOptimizedGridAxesChecked.Interface
open VeriTile.Triton FP.Structural BmmOptimized
noncomputable section

theorem real_complete (s : Shape) (c : Config) (rest : Algebra ℝ)
    (ps : List Program) (order : ps.Perm (programs s c)) :
    (realIO s c rest ps).Implements (realFormula s) := by
  have target : scalarFormula s realScalar=realFormula s := by
    funext vals o
    have ho : o=(0 : Fin 1) := Fin.eq_zero o
    subst o
    funext j
    change ordered s realScalar (arrays 0 s vals) (outputCell s j.1)=
      mathematical s (arrays 0 s vals) (outputCell s j.1)
    exact ordered_real s (arrays 0 s vals) (outputCell s j.1)
  simpa only [realIO,target] using complete s c realScalar rest ps order

theorem native_scalar (actual target : Native.Instructions)
    (fma : ∀ l r carry,actual.fmaRN l r carry=target.fmaRN l r carry) :
    Native.scalar actual=Native.scalar target := by
  change Scalar.mk Native.zero actual.fmaRN=Scalar.mk Native.zero target.fmaRN
  congr 1
  funext l r carry
  exact fma l r carry

theorem native_complete (s : Shape) (c : Config) (actual target : Native.Instructions)
    (fma : ∀ l r carry,actual.fmaRN l r carry=target.fmaRN l r carry)
    (rest : Algebra Native.Word) (ps : List Program) (order : ps.Perm (programs s c)) :
    (nativeIO s c actual rest ps).Implements (scalarFormula s (Native.scalar target)) := by
  have h:=complete s c (Native.scalar actual) rest ps order
  change (nativeIO s c actual rest ps).Implements (scalarFormula s (Native.scalar actual)) at h
  simpa only [native_scalar actual target fma] using h

#axiomsClean real_complete
#axiomsClean native_complete
end
end VeriTile.Bench.Repairs.BmmOptimizedGridAxesChecked.Interface
