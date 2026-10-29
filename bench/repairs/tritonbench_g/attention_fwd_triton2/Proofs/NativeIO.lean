import bench.repairs.tritonbench_g.attention_fwd_triton2.Proofs.IOResults

namespace VeriTile.Bench.Repairs.AttentionFwdTriton2.NativeIO
open VeriTile Triton FP.Structural Whole001 Source Whole Machine Interface
noncomputable section

theorem observes (I : KernelOnly.Instructions) (p : Parameters) (bases : Bases) (dom : Domain p bases)
    (sep : p.Out≠p.Q ∧ p.Out≠p.K ∧ p.Out≠p.V ∧ p.Out≠p.QS ∧ p.Out≠p.KS)
    (work : List Program) (order : (programs (grid p)).Perm work) :
    (io Native.algebra Native.dotModel p nativeInput (fun j=>observe32 I j.2.2.2.1.val) work).Observes
      (fun _ _ word=>word.setWidth 16) (nativeFormula I p) := by
  let selected := io Native.algebra Native.dotModel p nativeInput (fun j=>observe32 I j.2.2.2.1.val) work
  have target : nativeFormula I p=fun vals=>selected.mapOutputs (fun _ _ word=>word.setWidth 16)
      (IOResults.formula Native.algebra Native.dotModel p nativeInput (fun j=>observe32 I j.2.2.2.1.val) vals) := by
    funext vals o j
    have ho : o=(0 : Fin 1) := Fin.eq_zero o
    subst o
    rfl
  rw [target]
  exact ProgramIO.Observes.of_implements
    (IOResults.implements Native.algebra Native.dotModel p bases dom sep nativeInput
      (fun j=>observe32 I j.2.2.2.1.val) work order) (fun _ _ word=>word.setWidth 16)

#axiomsClean observes
end
end VeriTile.Bench.Repairs.AttentionFwdTriton2.NativeIO
