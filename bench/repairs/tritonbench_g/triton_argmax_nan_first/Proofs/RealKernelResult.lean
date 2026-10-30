import bench.repairs.tritonbench_g.triton_argmax_nan_first.Proofs.TypedRealGrid
import bench.repairs.tritonbench_g.triton_argmax_nan_first.Contract
import bench.repairs.tritonbench_g.triton_argmax_nan_first.Proofs.Request

namespace VeriTile.Bench.Repairs.TritonArgmaxNanFirst.RealKernelResult
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.TritonArgmax Source
noncomputable section

/-- Complete two-stage shared-quotation Real execution, with scalar precision
casts erased. The input is an arbitrary logical Real sequence. -/
theorem global (kind : TypedScratch.Kind) (rest : Algebra Math.Value) (s : State Math.Value)
    (X V : RegionName) (I O : Region .int) (inputValue : X≠V) (inputIndex : X≠Region.cast I)
    (scratchApart : V≠Region.cast I) (c : GlobalConfig) (coverage : Math.GlobalCoverage c)
    (input : Nat → ℝ) (index64 : Bool)
    (reads : ∀ i,i<c.size → (s.mem X i).read .real=(input i : Math.Value)) :
    ∃ middle final,
      TypedReal.GridRun (TypedReal.backend rest) (TypedScratch.firstKernel kind X V I c.size c.block 0 index64)
        (List.range c.blocks) s middle ∧
      FP.Structural.exec (TypedReal.backend rest) (TypedScratch.secondKernel kind V I O c.blocks c.midBlock) middle=some final ∧
      final.mem (Region.cast O) 0=.mk .int (Int.ofNat
        (Math.argmax c.size coverage.sizePositive (fun i : Fin c.size => input i.val)).val) ∧
      (∀ region offset,(region≠Region.cast O ∨ offset≠0) →
        (∀ pid,pid<c.blocks → (region≠V ∨ offset≠pid) ∧ (region≠Region.cast I ∨ offset≠pid)) →
        final.mem region offset=s.mem region offset) :=
  TypedReal.global_source_reference kind rest s X V I O inputValue inputIndex scratchApart c coverage input index64 reads

theorem dimension (rest : Algebra Math.Value) (s : State Math.Value) (X : RegionName) (O : Region .int)
    (M N K BM BN base : Nat) (index64 : Bool) (positive : 0<BN) (nonempty : 0<N) (innerPositive : 0<K)
    (input : Nat → Nat → Nat → ℝ)
    (reads : ∀ row col inner,row<M → col<N → inner<K →
      (s.mem X (row*N*K+col*K+inner)).read .real=(input row inner col : Math.Value)) :
    ∃ t, FP.Structural.exec (TypedReal.backend rest) (TypedSource.dimension X O M N K BM BN base index64) s=some t ∧
      (∀ i : Fin BM,((s.pids 0+base)/K)*BM+i.val<M →
        t.mem (Region.cast O) ((((s.pids 0+base)/K)*BM+i.val)*K+(s.pids 0+base)%K)=
          .mk .int (Int.ofNat ((Math.argmax N nonempty (fun j : Fin N =>
            input (((s.pids 0+base)/K)*BM+i.val) ((s.pids 0+base)%K) j.val)).val))) ∧
      (∀ region offset,(region≠Region.cast O ∨
        ∀ i : Fin BM,((s.pids 0+base)/K)*BM+i.val<M →
          offset≠((((s.pids 0+base)/K)*BM+i.val)*K+(s.pids 0+base)%K)) →
        t.mem region offset=s.mem region offset) :=
  TypedReal.source_reference rest s X O M N K BM BN base index64 positive nonempty innerPositive input reads

/-- Integer/bool Real projection erases the entire precision macro at once.
The exact native macro and requested widths are retained by FPEquiv. This
states no equality between finite Real clamps and arbitrary Real inputs. -/
theorem integer_precision_erasure (kind : Formats.IntegerKind) (word : Math.Value) :
    (ConversionMacro.SaturatingConversion.mk kind).idealized word=word := rfl

/-- A floating request retains its quotation channel. Erasing an integer/bool
precision macro uses the Real value channel; its native width is still kept
by Request.Kind.format and FPEquiv.global. -/
def projectedKind : Request.Kind → TypedScratch.Kind
  | .floating kind => kind
  | .integer _ => .fp32

theorem requested_global (kind : Request.Kind) (rest : Algebra Math.Value) (s : State Math.Value)
    (X V : RegionName) (I O : Region .int) (inputValue : X≠V) (inputIndex : X≠Region.cast I)
    (scratchApart : V≠Region.cast I) (c : GlobalConfig) (coverage : Math.GlobalCoverage c)
    (input : Nat → ℝ) (index64 : Bool)
    (reads : ∀ i,i<c.size → (s.mem X i).read .real=(input i : Math.Value)) :
    ∃ middle final,
      TypedReal.GridRun (TypedReal.backend rest)
        (TypedScratch.firstKernel (projectedKind kind) X V I c.size c.block 0 index64)
        (List.range c.blocks) s middle ∧
      FP.Structural.exec (TypedReal.backend rest)
        (TypedScratch.secondKernel (projectedKind kind) V I O c.blocks c.midBlock) middle=some final ∧
      final.mem (Region.cast O) 0=.mk .int (Int.ofNat
        (Math.argmax c.size coverage.sizePositive (fun i : Fin c.size => input i.val)).val) ∧
      (∀ region offset,(region≠Region.cast O ∨ offset≠0) →
        (∀ pid,pid<c.blocks → (region≠V ∨ offset≠pid) ∧ (region≠Region.cast I ∨ offset≠pid)) →
        final.mem region offset=s.mem region offset) :=
  global (projectedKind kind) rest s X V I O inputValue inputIndex scratchApart c coverage input index64 reads

#axiomsClean global
#axiomsClean dimension
#axiomsClean integer_precision_erasure
#axiomsClean requested_global
#specNonCircular Math.argmax avoiding [TypedReal.vectorLoop,TypedReal.GridRun,RealScratch.run]
end
end VeriTile.Bench.Repairs.TritonArgmaxNanFirst.RealKernelResult
