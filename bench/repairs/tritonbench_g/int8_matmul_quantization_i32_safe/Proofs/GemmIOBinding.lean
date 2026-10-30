import bench.repairs.tritonbench_g.int8_matmul_quantization_i32_safe.Proofs.GemmIOGeometry

namespace VeriTile.Bench.Repairs.Int8MatmulQuantization.I32Safe.PublicIO.Gemm
open VeriTile.Triton FP.Structural KernelExecution
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 28000
variable {F I H : Type} [Inhabited F]

theorem reads (zero : F) (s : Shape) (c : Config) (rowScale output : Precision)
    (columnMajor : Bool) (xs : Inputs F s rowScale) (base : State F)
    (signed : SignedInputs s rowScale xs)
    (binding : (layout F s rowScale output columnMajor).InputCells id xs base) :
    I32Safe.Gemm.Reads (I32Safe.Gemm.parameters s c columnMajor) rowScale
      (input zero s rowScale xs) base := by
  constructor
  · intro r k hr hk
    change r<s.M at hr
    change k<s.K at hk
    have h:=congrArg (FP.Structural.Cell.read .int) (binding 0 (⟨r,hr⟩,⟨k,hk⟩,()) trivial)
    change (base.mem "A" (r*max s.K 1+k)).read .int=xs 0 (⟨r,hr⟩,⟨k,hk⟩,()) at h
    simp only [I32Safe.Gemm.parameters,Nat.mul_one]
    change (base.mem "A" (r*max s.K 1+k)).read .int=
      ((input zero s rowScale xs).A r k).toInt
    simpa only [input,dif_pos hr,dif_pos hk,signed.1] using h
  · intro k n hk hn
    change k<s.K at hk
    change n<s.N at hn
    have h:=congrArg (FP.Structural.Cell.read .int) (binding 1 (⟨k,hk⟩,⟨n,hn⟩,()) trivial)
    change (base.mem "B" (if columnMajor then n*max s.K 1+k else k*max s.N 1+n)).read .int=
      xs 1 (⟨k,hk⟩,⟨n,hn⟩,()) at h
    change (base.mem "B" (k*(if columnMajor then 1 else max s.N 1)+n*(if columnMajor then max s.K 1 else 1))).read .int=
      ((input zero s rowScale xs).B k n).toInt
    cases columnMajor <;> simpa only [input,dif_pos hk,dif_pos hn,signed.2,Bool.false_eq_true,
      if_false,if_true,Nat.mul_one,Nat.add_comm] using h
  · intro r hr
    change r<s.M at hr
    have h:=binding 2 (⟨r,hr⟩,()) trivial
    cases rowScale
    · have h':=congrArg (FP.Structural.Cell.read .fp16) h
      change (base.mem "AS" r).read .fp16=xs 2 (⟨r,hr⟩,()) at h'
      simp only [I32Safe.Gemm.parameters,Nat.mul_one]
      change (base.mem "AS" r).read .fp16=(input zero s .half xs).aScale r
      simpa only [input,dif_pos hr,fromValue] using h'
    · have h':=congrArg (FP.Structural.Cell.read .real) h
      change (base.mem "AS" r).read .real=xs 2 (⟨r,hr⟩,()) at h'
      simp only [I32Safe.Gemm.parameters,Nat.mul_one]
      change (base.mem "AS" r).read .real=(input zero s .single xs).aScale r
      simpa only [input,dif_pos hr,fromValue] using h'
  · intro n hn
    change n<s.N at hn
    have h:=congrArg (FP.Structural.Cell.read .real) (binding 3 (⟨n,hn⟩,()) trivial)
    change (base.mem "BS" n).read .real=xs 3 (⟨n,hn⟩,()) at h
    simp only [I32Safe.Gemm.parameters,Nat.mul_one]
    change (base.mem "BS" n).read .real=(input zero s rowScale xs).bScale n
    simpa only [input,dif_pos hn] using h

theorem outside_exterior (s : Shape) (rowScale output : Precision) (columnMajor : Bool)
    (xs : Inputs F s rowScale) (r : RegionName) (off : Nat)
    (outside : (layout F s rowScale output columnMajor).Outside xs r off) :
    r≠"C" ∨ s.M*s.N≤off := by
  by_cases hr : r="C"
  · right
    by_contra large
    have hbound : off<s.M*s.N:=Nat.lt_of_not_ge large
    have hn : 0<s.N:=by nlinarith
    have row : off/s.N<s.M:=(Nat.div_lt_iff_lt_mul hn).mpr hbound
    have col : off%s.N<s.N:=Nat.mod_lt _ hn
    have bad:=outside.1 0 (⟨off/s.N,row⟩,⟨off%s.N,col⟩,()) trivial
    have address : off=off/s.N*s.N+off%s.N:=by
      have h:=Nat.mod_add_div off s.N
      nlinarith
    exact bad.elim (fun h=>h hr) (fun h=>h address)
  · exact Or.inl hr

theorem exterior_outside (s : Shape) (rowScale output : Precision) (columnMajor : Bool)
    (xs : Inputs F s rowScale) (r : RegionName) (off : Nat)
    (outside : r≠"C" ∨ s.M*s.N≤off) :
    (layout F s rowScale output columnMajor).Outside xs r off := by
  constructor
  · intro port j _
    have hp : port=0:=Fin.eq_zero port
    subst port
    rcases outside with hr | ho
    · exact Or.inl hr
    · right
      have bound:=RowwiseQuantizationTriton.linear_bound s.M s.N j.1 j.2.1 j.1.isLt j.2.1.isLt
      change off≠j.1.val*s.N+j.2.1.val
      omega
  · intro port
    exact port.elim0

theorem patch_other (s : Shape) (rowScale output : Precision) (columnMajor : Bool)
    (xs : Inputs F s rowScale) (current : (layout F s rowScale output columnMajor).RawOutputs F)
    (base : State F) (r : RegionName) (off : Nat) (different : r≠"C") :
    ((layout F s rowScale output columnMajor).patchOutputs xs current base).mem r off=base.mem r off :=
  (layout F s rowScale output columnMajor).patchOutputs_frame xs current base r off
    (exterior_outside s rowScale output columnMajor xs r off (Or.inl different))

theorem patch_inputs (s : Shape) (rowScale output : Precision) (columnMajor : Bool)
    (xs : Inputs F s rowScale) (current : (layout F s rowScale output columnMajor).RawOutputs F)
    (base : State F) (binding : (layout F s rowScale output columnMajor).InputCells id xs base) :
    (layout F s rowScale output columnMajor).InputCells id xs
      ((layout F s rowScale output columnMajor).patchOutputs xs current base) := by
  intro port j active
  have other : (layout F s rowScale output columnMajor).inputs[port].buf≠"C" := by
    rcases port with ⟨port,hp⟩
    have small : port<4:=hp
    interval_cases port <;> simp [layout,inputs]
  rw [patch_other s rowScale output columnMajor xs current base _ _ other]
  exact binding port j active

theorem patch_cell (s : Shape) (rowScale output : Precision) (columnMajor : Bool)
    (xs : Inputs F s rowScale) (current : (layout F s rowScale output columnMajor).RawOutputs F)
    (base : State F) (cell : MatrixCell s) :
    ((layout F s rowScale output columnMajor).patchOutputs xs current base).mem "C" (Grid.outputOffset s cell)=
      FP.Structural.Cell.mk (dtype output) (current ⟨0,(cell.1,cell.2,())⟩) :=
  (layout F s rowScale output columnMajor).patchOutputs_read xs
    (output_unique F s rowScale output columnMajor xs) current base ⟨0,(cell.1,cell.2,())⟩ trivial

/-- The finite typed initializer is exactly the original C-only reset. -/
theorem patch_reset (zero : F) (s : Shape) (rowScale output : Precision) (columnMajor : Bool)
    (xs : Inputs F s rowScale) (base : State F) :
    (layout F s rowScale output columnMajor).patchOutputs xs (initialValues zero s output) base=
      Grid.reset s (Grid.zeroCell output zero) base := by
  have hm : ((layout F s rowScale output columnMajor).patchOutputs xs (initialValues zero s output) base).mem=
      (Grid.reset s (Grid.zeroCell output zero) base).mem := by
    funext r off
    by_cases active : r="C" ∧ off<s.M*s.N
    · obtain ⟨rfl,bound⟩:=active
      have hn : 0<s.N:=by nlinarith
      let cell : MatrixCell s:=⟨⟨off/s.N,(Nat.div_lt_iff_lt_mul hn).mpr bound⟩,⟨off%s.N,Nat.mod_lt _ hn⟩⟩
      have address : Grid.outputOffset s cell=off:=by
        dsimp [Grid.outputOffset,cell]
        have h:=Nat.mod_add_div off s.N
        nlinarith
      rw [←address,patch_cell,Grid.reset_cell]
      cases output <;> rfl
    · have outside : r≠"C" ∨ s.M*s.N≤off:=by
        by_cases hr : r="C"
        · exact Or.inr (Nat.le_of_not_lt (fun h=>active ⟨hr,h⟩))
        · exact Or.inl hr
      rw [(layout F s rowScale output columnMajor).patchOutputs_frame xs (initialValues zero s output) base r off
        (exterior_outside s rowScale output columnMajor xs r off outside),Grid.reset_frame _ _ _ _ _ outside]
  cases base
  exact congrArg (fun memory=>State.mk memory _ _ _ _) hm

/-- Original launch setup only installs PID axes 0 and 1. -/
theorem initial_eq (M : Algebra F) (carrier : IntegerCarrier I) (ops : Int8DequantMatmul.Ops I F H)
    (zero : carrier.decode ops.izero=0) (plans : Plans) (rowScale output : Precision)
    (s : Shape) (c : Config) (columnMajor : Bool) (p : Program s c) (base : State F) :
    (invocation M carrier ops zero plans rowScale output s c columnMajor p).initial base=
      Grid.atProgram base p.1 p.2 := by
  simp [IntegerDotInvocation.initial,invocation,Invocation.initial,bindRegisters,Grid.atProgram]
  funext ax
  by_cases h0 : ax=0 <;> by_cases h1 : ax=1 <;> simp_all

theorem invocation_run (M : Algebra F) (carrier : IntegerCarrier I) (ops : Int8DequantMatmul.Ops I F H)
    (zero : carrier.decode ops.izero=0) (plans : Plans) (rowScale output : Precision)
    (s : Shape) (c : Config) (columnMajor : Bool) (p : Program s c) (base : State F) :
    (invocation M carrier ops zero plans rowScale output s c columnMajor p).run base=
      KernelExecution.exec M carrier ops plans (I32Safe.Gemm.source rowScale output (I32Safe.Gemm.parameters s c columnMajor))
        (Grid.atProgram base p.1 p.2) := by
  unfold IntegerDotInvocation.run
  rw [initial_eq]
  exact SharedInteger.exec_eq M carrier ops zero plans _ _

#axiomsClean reads
#axiomsClean patch_inputs
#axiomsClean patch_reset
#axiomsClean invocation_run
end
end VeriTile.Bench.Repairs.Int8MatmulQuantization.I32Safe.PublicIO.Gemm
