import bench.repairs.tritonbench_g.int8_matmul_quantization_i32_safe.Proofs.KernelInterface

namespace VeriTile.Bench.Repairs.Int8MatmulQuantization.I32Safe.Public
open VeriTile.Triton FP.Structural KernelExecution
open KernelProducerGrid
noncomputable section

local instance kernelInhabitedWord32 : Inhabited Word32 := ⟨FP.Value.zero .fp32⟩

theorem fpEquiv (f : KernelNativeModel.FloatingInstructions) (tree : KernelScalarProfile.Tree)
    (rest : Algebra Word32) (plans : Plans) (rowScale output : Precision) (s : Shape) (c : Config)
    (columnMajor : Bool) (input : Gemm.Input Word32) (q : Int8MatmulQuantization.Schedule s c)
    (initial : State Word32) (reads : Gemm.Reads (Gemm.parameters s c columnMajor) rowScale input initial) :
    KernelInterface.NativeResult f tree rest plans rowScale output s c columnMajor input q initial :=
  KernelInterface.native f tree rest plans rowScale output s c columnMajor input q initial reads

theorem producerFPEquiv (f : KernelNativeModel.FloatingInstructions) (p : Geometry)
    (selection : Producer.Selection p) (covers : p.columns ≤ p.width) (rest : Algebra Word32)
    (values : Nat → Nat → Word32) (s : State Word32)
    (reads : KernelProducerTargets.Reads p values s) (order : Producer.Schedule p) :
    ∃ final, KernelProducerGrid.run (KernelScalarProfile.native f selection.tree rest) KernelNativeModel.integer8 p order.order s = some final ∧
      (∀ r : Fin p.rows, ∀ c : Fin p.width, c.val < p.columns →
        final.mem "q" (r.val*p.columns+c.val) = .mk .int (KernelProducerTargets.nativeByte f p selection (values r) c).toInt) ∧
      (∀ r : Fin p.rows, final.mem "scales" r =
        .mk .fp16 (KernelScalarProfile.pack16 (KernelProducerTargets.nativeStoredScale f p selection (values r)))) ∧
      (∀ (region : RegionName) offset, (region ≠ "q" ∨ p.rows*p.columns ≤ offset) →
        (region ≠ "scales" ∨ p.rows ≤ offset) → final.mem region offset = s.mem region offset) :=
  Producer.native_complete f p selection covers rest values s reads order

theorem composedFPEquiv (f : KernelNativeModel.FloatingInstructions) (s : Shape) (c : Config)
    (columnMajor : Bool) (width : Nat) (positive : 0<width) (covers : s.K≤width)
    (selection : Producer.Selection (Composed.geometry s width positive)) (rest : Algebra Word32) (plans : Plans)
    (output : Precision) (values : Nat → Nat → Word32) (B : Nat → Nat → Byte) (bs : Nat → Word32)
    (state : State Word32) (reads : KernelProducerTargets.Reads (Composed.geometry s width positive) values state)
    (weights : Composed.WeightReads s c columnMajor B bs state)
    (order : Producer.Schedule (Composed.geometry s width positive)) (schedule : Int8MatmulQuantization.Schedule s c) :
    ∃ produced,
      KernelProducerGrid.run (KernelScalarProfile.native f selection.tree rest) KernelNativeModel.integer8
        (Composed.geometry s width positive) order.order state=some produced ∧
      (∀ r : Fin s.M, ∀ k : Fin width, k.val<s.K → produced.mem "q" (r.val*s.K+k.val)=
        .mk .int (KernelProducerTargets.nativeByte f (Composed.geometry s width positive) selection (values r) k).toInt) ∧
      (∀ r : Fin s.M, produced.mem "scales" r=.mk .fp16
        (KernelScalarProfile.pack16 (KernelProducerTargets.nativeStoredScale f (Composed.geometry s width positive) selection (values r)))) ∧
      (∀ (region : RegionName) offset, (region≠"q" ∨ s.M*s.K≤offset) →
        (region≠"scales" ∨ s.M≤offset) → produced.mem region offset=state.mem region offset) ∧
      KernelInterface.NativeResult f selection.tree rest plans .half output s c columnMajor
        (Composed.nativeInput f (Composed.geometry s width positive) selection values B bs) schedule (Composed.attach produced) := by
  obtain ⟨produced,run,data,scales,frame⟩ := producerFPEquiv f (Composed.geometry s width positive)
    selection covers rest values state reads order
  have bound : Gemm.Reads (Gemm.parameters s c columnMajor) .half
      (Composed.nativeInput f (Composed.geometry s width positive) selection values B bs) (Composed.attach produced) :=
    Composed.reads_produced s c columnMajor width covers state produced _ _ _ _ data scales weights
      (fun region i hq hs => frame region i (Or.inl hq) (Or.inl hs))
  exact ⟨produced,run,data,scales,frame,fpEquiv f selection.tree rest plans .half output s c columnMajor
    (Composed.nativeInput f (Composed.geometry s width positive) selection values B bs) schedule (Composed.attach produced) bound⟩

#axiomsClean fpEquiv
#axiomsClean producerFPEquiv
#axiomsClean composedFPEquiv
end
end VeriTile.Bench.Repairs.Int8MatmulQuantization.I32Safe.Public
