import bench.tritonbench_g.int8_matmul_quantization.Proofs.GemmIOSafety
import bench.tritonbench_g.int8_matmul_quantization.Proofs.GemmIOBinding

namespace VeriTile.Bench.Repairs.Int8MatmulQuantization.I32Safe.PublicIO.GemmSafety
open VeriTile.Triton FP.Structural KernelExecution KernelGemm
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 32000
variable {F I H : Type} [Inhabited F]

theorem range_safe (M : Algebra F) (carrier : IntegerCarrier I) (ops : Int8DequantMatmul.Ops I F H)
    (zero : carrier.decode ops.izero=0) (plans : Plans) (rowScale : Precision)
    (p : Parameters) (initial current : State F) (t : Nat) (ht : t≤p.steps)
    (h : Invariant M carrier ops plans rowScale p initial t current)
    (bounds : RegionName→Nat) (hb : Bounds p initial bounds) :
    IntegerDot.rangeSafe M (SharedInteger.arithmetic carrier ops zero plans) (SharedInteger.context plans)
      bounds "k" t p.steps 1 (body p) current := by
  have aux : ∀ n t (current : State F),p.steps-t=n → t≤p.steps →
      Invariant M carrier ops plans rowScale p initial t current →
      IntegerDot.rangeSafe M (SharedInteger.arithmetic carrier ops zero plans) (SharedInteger.context plans)
        bounds "k" t p.steps 1 (body p) current := by
    intro n
    induction n using Nat.strong_induction_on with
    | h n ih=>
      intro t current hn ht invariant
      rw [IntegerDot.rangeSafe_unfold]
      by_cases active : t<p.steps
      · rw [if_pos ⟨by decide,active⟩]
        refine ⟨body_safe M carrier ops zero plans rowScale p initial current t invariant bounds hb,?_⟩
        rw [SharedInteger.run_eq,(bodyExecution M carrier ops plans rowScale p initial current t invariant).property]
        exact ih (p.steps-(t+1)) (by omega) (t+1) _ rfl (by omega)
          (bodyInvariant M carrier ops plans rowScale p initial current t invariant)
      · rw [if_neg (by simpa using active)]
        trivial
  exact aux _ t current rfl ht h

/-- Complete original source, including every actual K-loop successor and
both final branches. No access-safety premise enters a public numerical law. -/
theorem source_safe (M : Algebra F) (carrier : IntegerCarrier I) (ops : Int8DequantMatmul.Ops I F H)
    (zero : carrier.decode ops.izero=0) (plans : Plans) (rowScale output : Precision)
    (p : Parameters) (initial : State F) (bounds : RegionName→Nat) (hb : Bounds p initial bounds) :
    IntegerDot.TraceSafe M (SharedInteger.arithmetic carrier ops zero plans) (SharedInteger.context plans)
      bounds (I32Safe.Gemm.source rowScale output p) initial := by
  unfold IntegerDot.TraceSafe
  rw [I32Safe.Gemm.source_identity,source_decomposition,List.append_assoc,IntegerDot.traceSafeList_append]
  refine ⟨prelude_safe M carrier ops zero plans rowScale output p initial bounds hb,?_⟩
  intro base ran
  rw [SharedInteger.run_eq,(preludeExecution M carrier ops plans rowScale output p initial).property] at ran
  cases Option.some.inj ran
  have inv:=preludeInvariant M carrier ops plans rowScale output p initial
  rw [List.singleton_append,IntegerDot.traceSafeList_cons]
  constructor
  · rw [IntegerDot.stepSafe_unfold]
    exact range_safe M carrier ops zero plans rowScale p initial _ 0 (Nat.zero_le _) inv bounds hb
  · intro final ran
    simp only [IntegerDot.step_unfold] at ran
    rw [SharedInteger.range_eq] at ran
    obtain ⟨mid,executed,invariant⟩:=loopExecution M carrier ops plans rowScale p initial _ 0 (Nat.zero_le _) inv
    rw [executed] at ran
    cases Option.some.inj ran
    exact epilogue_safe M carrier ops zero plans rowScale output p initial _ invariant bounds hb

theorem bounds_of_io (s : Shape) (c : Config) (rowScale output : Precision) (columnMajor : Bool)
    (xs : Gemm.Inputs F s rowScale) (bounds : RegionName→Nat)
    (bounded : (Gemm.layout F s rowScale output columnMajor).InBounds bounds xs)
    (p : Gemm.Program s c) (base : State F) :
    Bounds (I32Safe.Gemm.parameters s c columnMajor) (Grid.atProgram base p.1 p.2) bounds := by
  let param:=I32Safe.Gemm.parameters s c columnMajor
  let initial:=Grid.atProgram base p.1 p.2
  have rows : 0<s.M:=Gemm.rows_pos s c p.1 p.1.isLt
  have cols : 0<s.N:=Gemm.columns_pos s c p.1 p.1.isLt
  have rowBound (r : Fin c.BM) : inputRow param initial r<s.M:=Nat.mod_lt _ rows
  have columnBound (n : Fin c.BN) : inputColumn param initial n<s.N:=Nat.mod_lt _ cols
  constructor
  · intro t i mask
    let k:=inputInner param initial i.2.1+t*(c.BK*c.S)
    have hk : k<s.K:=by change inputInner param initial i.2.1<s.K-t*(c.BK*c.S) at mask;dsimp [k];omega
    have h:=bounded.2.1 0 (⟨inputRow param initial i.1,rowBound i.1⟩,⟨k,hk⟩,()) trivial
    change inputRow param initial i.1*max s.K 1+k<bounds "A" at h
    change aOffset param initial t i<bounds "A"
    simpa only [aOffset,param,I32Safe.Gemm.parameters,Nat.mul_one,k,Nat.add_assoc] using h
  · intro t i mask
    let k:=inputInner param initial i.1+t*(c.BK*c.S)
    have hk : k<s.K:=by change inputInner param initial i.1<s.K-t*(c.BK*c.S) at mask;dsimp [k];omega
    have h:=bounded.2.1 1 (⟨k,hk⟩,⟨inputColumn param initial i.2.1,columnBound i.2.1⟩,()) trivial
    change (if columnMajor then inputColumn param initial i.2.1*max s.K 1+k else k*max s.N 1+inputColumn param initial i.2.1)<bounds "B" at h
    change bOffset param initial t i<bounds "B"
    convert h using 1
    cases columnMajor <;> simp [bOffset,param,I32Safe.Gemm.parameters,k] <;> ring
  · intro r _mask
    simpa only [param,I32Safe.Gemm.parameters,Nat.mul_one] using
      bounded.2.1 2 (⟨inputRow param initial r,rowBound r⟩,()) trivial
  · intro n _mask
    simpa only [param,I32Safe.Gemm.parameters,Nat.mul_one] using
      bounded.2.1 3 (⟨inputColumn param initial n,columnBound n⟩,()) trivial
  · intro i mask
    have h:=bounded.2.2.1 0
      (⟨rowBlock param initial*c.BM+i.1.val,mask.1⟩,⟨columnBlock param initial*c.BN+i.2.1.val,mask.2⟩,()) trivial
    change (rowBlock param initial*c.BM+i.1.val)*s.N+(columnBlock param initial*c.BN+i.2.1.val)<bounds "C" at h
    change outputOffset param initial i<bounds "C"
    change max s.N 1*(rowBlock param initial*c.BM+i.1.val)+1*(columnBlock param initial*c.BN+i.2.1.val)<bounds "C"
    have maxN : max s.N 1=s.N := by omega
    rw [maxN,Nat.one_mul,Nat.mul_comm s.N]
    exact h

/-- Source safety of every complete invocation, for arbitrary memory values. -/
theorem invocation_safe (M : Algebra F) (carrier : IntegerCarrier I) (ops : Int8DequantMatmul.Ops I F H)
    (zero : carrier.decode ops.izero=0) (plans : Plans) (rowScale output : Precision)
    (s : Shape) (c : Config) (columnMajor : Bool) (p : Gemm.Program s c)
    (xs : Gemm.Inputs F s rowScale) (bounds : RegionName→Nat)
    (bounded : (Gemm.layout F s rowScale output columnMajor).InBounds bounds xs) (base : State F) :
    (Gemm.invocation M carrier ops zero plans rowScale output s c columnMajor p).Safe bounds base := by
  unfold IntegerDotInvocation.Safe
  rw [Gemm.initial_eq]
  exact source_safe M carrier ops zero plans rowScale output _ _ bounds
    (bounds_of_io s c rowScale output columnMajor xs bounds bounded p base)

#axiomsClean source_safe
#axiomsClean bounds_of_io
#axiomsClean invocation_safe
end
end VeriTile.Bench.Repairs.Int8MatmulQuantization.I32Safe.PublicIO.GemmSafety
