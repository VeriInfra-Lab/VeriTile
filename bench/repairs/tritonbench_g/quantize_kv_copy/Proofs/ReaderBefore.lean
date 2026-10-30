import bench.repairs.tritonbench_g.quantize_kv_copy.Proofs.IOGrid

namespace VeriTile.Bench.Repairs.QuantizeKvCopy.ReaderBefore
open VeriTile.Triton FP.Structural
open Kernel KernelModels
open scoped VeriTile.Triton.FP.Structural.ProgramIO
noncomputable section

/-- IO.lean exposes all input groups, the signed destination metadata, both
INT8 code / FP32 scale outputs and the entire token/head source schedule. -/
def quantizeIO (p : Parameters) (n : Nat) (dest : Nat → Nat) : ProgramIO ℝ ℝ :=
  PublicIO.sourceIO KernelModels.realOps p n dest

/- The reused mathematicalMaximum is the shared logical maximum with
M(x,0)=0 and M(x,n+1)=max (M(x,n)) |x n|. Thus its extent below is the
logical row extent; no execution or padded reduction is part of this formula. -/

/-- Each independent eight-element group has scale max(abs(x))/127 and the
original toward-zero INT8 code formula, including the explicit all-zero case.
Both complete outputs and the exact scatter frame are part of the IO judgment. -/
theorem mathematical (p : Parameters) (n : Nat) (dest : Nat → Nat)
    (covers : p.groups ≤ p.block)
    (unique : ∀ a b, a<n → b<n →
      KernelGridResult.target p dest a = KernelGridResult.target p dest b → a=b) :
    Spec.Real (quantizeIO p n dest ⊨{PublicIO.Metadata dest}
      (fun values => PublicIO.result
        (fun pid g c =>
          let x := PublicIO.elements values pid g
          ((if mathematicalMaximum x 8=0 then (0 : BitVec 8) else
            let z := x c/(mathematicalMaximum x 8/127)
            BitVec.ofInt 8 (if z<0 then -Int.floor (-z) else Int.floor z))).toInt)
        (fun pid g => mathematicalMaximum (PublicIO.elements values pid g) 8/127))) := by
  apply PublicIO.implements
  intro values s reads
  exact KernelRaw.mathematical p n dest covers unique (PublicIO.elements values) s reads

end
end VeriTile.Bench.Repairs.QuantizeKvCopy.ReaderBefore
