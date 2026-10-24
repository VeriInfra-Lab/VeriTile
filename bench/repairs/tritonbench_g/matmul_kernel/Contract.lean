import bench.repairs.tritonbench_g.matmul_kernel.Proofs.Admitted

local macro "true" : term => `(Bool.true)

namespace VeriTile.Bench.Repairs.MatmulKernel.Checked

def blockCheck (b : Nat) : Prop := 16 ≤ b ∧ b &&& (b-1) = 0
instance (b : Nat) : Decidable (blockCheck b) := inferInstanceAs (Decidable (_ ∧ _))

def viewCheck (v : View) : Prop := 0 < v.sr ∧ 0 < v.sc ∧
  (v.rows = 0 ∨ v.cols = 0 ∨ v.cols*v.sc ≤ v.sr ∨ v.rows*v.sr ≤ v.sc)
instance (v : View) : Decidable (viewCheck v) := inferInstanceAs (Decidable (_ ∧ _ ∧ _))

def configChecks (c : Config) : Bool :=
  decide (blockCheck c.BM ∧ blockCheck c.BN ∧ blockCheck c.BK ∧
    c.A.rows = c.M ∧ c.A.cols = c.K ∧ c.B.rows = c.K ∧ c.B.cols = c.N ∧
    c.C.rows = c.M ∧ c.C.cols = c.N ∧ viewCheck c.A ∧ viewCheck c.B ∧ viewCheck c.C)

def indexChecks (c : Config) : Bool :=
  decide (c.M = 0 ∨ c.N = 0 ∨
    (c.M < 2^31 ∧ c.N < 2^31 ∧ c.K < 2^31 ∧
    c.BM < 2^31 ∧ c.BN < 2^31 ∧ c.BK < 2^31 ∧
    ceil c.M c.BM*c.BM < 2^31 ∧ ceil c.N c.BN*c.BN < 2^31 ∧ c.steps*c.BK+c.BK < 2^31 ∧
    c.A.sr < 2^31 ∧ c.A.sc < 2^31 ∧ c.B.sr < 2^31 ∧ c.B.sc < 2^31 ∧ c.C.sr < 2^31 ∧ c.C.sc < 2^31 ∧
    (c.M-1)*c.A.sr+(c.steps*c.BK+c.BK)*c.A.sc < 2^31 ∧
    (c.steps*c.BK+c.BK)*c.B.sr+(c.N-1)*c.B.sc < 2^31 ∧
    (ceil c.M c.BM*c.BM)*c.C.sr+(ceil c.N c.BN*c.BN)*c.C.sc < 2^31))

structure Metadata where
  config : Config
  deviceA : Nat
  deviceB : Nat
  deviceC : Nat
  storageA : Nat
  storageB : Nat
  storageC : Nat
  pointerA : Nat
  pointerB : Nat
  pointerC : Nat
  resolvedA : Bool
  resolvedB : Bool
  resolvedC : Bool

def metadataChecks (p : Metadata) : Bool :=
  decide (p.deviceA = p.deviceB ∧ p.deviceA = p.deviceC ∧
    p.resolvedA = true ∧ p.resolvedB = true ∧ p.resolvedC = true ∧
    (0 < p.config.M*p.config.N → 0 < p.config.M*p.config.K → p.storageC ≠ p.storageA) ∧
    (0 < p.config.M*p.config.N → 0 < p.config.K*p.config.N → p.storageC ≠ p.storageB) ∧
    (p.config.M*p.config.K=0 ∨ p.pointerA%4=0) ∧
    (p.config.K*p.config.N=0 ∨ p.pointerB%4=0) ∧
    (p.config.M*p.config.N=0 ∨ p.pointerC%4=0))

inductive Error where | dynamic | metadata | configuration | indices deriving DecidableEq, Repr

def checkInputs (p : Metadata) : Except Error Unit :=
  if !metadataChecks p then .error .metadata
  else if !configChecks p.config then .error .configuration
  else if !indexChecks p.config then .error .indices
  else .ok ()

def InputContract (p : Metadata) : Prop := metadataChecks p = true ∧
  configChecks p.config = true ∧ indexChecks p.config = true

theorem checkInputs_iff (p : Metadata) : checkInputs p = .ok () ↔ InputContract p := by
  simp [checkInputs,InputContract,ite_eq_iff]

/-- Dynamic classification retains exact Python integer, dtype, CUDA, rank
and layout rejection before the representable ordinary Tensor descriptor. -/
inductive Arguments where
  | unsupported
  | supported (metadata : Metadata)

def checkArguments : Arguments → Except Error Unit
  | .unsupported => .error .dynamic
  | .supported p => checkInputs p

def AdmittedArguments : Arguments → Prop
  | .unsupported => False
  | .supported p => InputContract p

theorem checkArguments_iff (args : Arguments) :
    checkArguments args = .ok () ↔ AdmittedArguments args := by
  cases args <;> simp [checkArguments,AdmittedArguments,checkInputs_iff]

theorem indexChecks_iff (c : Config) :
    indexChecks c = true ↔ c.M = 0 ∨ c.N = 0 ∨ Signed32 c := by
  simp only [indexChecks,decide_eq_true_eq]
  constructor
  · rintro (hm|hn|⟨m,n,k,bm,bn,bk,pm,pn,pk,am,ak,bk',bn',cm,cn,af,bf,cf⟩)
    · exact Or.inl hm
    · exact Or.inr (Or.inl hn)
    · exact Or.inr (Or.inr ⟨⟨m,n,k⟩,⟨bm,bn,bk⟩,pm,pn,pk,⟨am,ak⟩,⟨bk',bn'⟩,⟨cm,cn⟩,af,bf,cf⟩)
  · rintro (hm|hn|h)
    · exact Or.inl hm
    · exact Or.inr (Or.inl hn)
    · exact Or.inr (Or.inr ⟨h.dims.1,h.dims.2.1,h.dims.2.2,h.blocks.1,h.blocks.2.1,h.blocks.2.2,
        h.paddedM,h.paddedN,h.paddedK,h.astrides.1,h.astrides.2,h.bstrides.1,h.bstrides.2,h.cstrides.1,h.cstrides.2,
        h.aformed,h.bformed,h.cformed⟩)

theorem block_atoms (b : Nat) (h : blockCheck b) : 0 < b ∧ b/8*8=b := by
  obtain ⟨e,he⟩ := (Nat.and_sub_one_eq_zero_iff_isPowerOfTwo (by have := h.1; omega)).1 h.2
  have ep : 4 ≤ e := by
    have hh : 2^4 ≤ 2^e := by simpa [←he] using h.1
    exact (Nat.pow_le_pow_iff_right (by decide : 1 < 2)).1 hh
  exact ⟨by have := h.1; omega, he ▸ power_of_two_atoms e ep⟩

/-- View origins are actual data pointers; capacity is the remaining span.
Nonzero storage offsets are therefore ordinary metadata, not an input ban. -/
def Allocation (v : View) : Prop :=
  v.rows = 0 ∨ v.cols = 0 ∨ (v.rows-1)*v.sr+(v.cols-1)*v.sc < v.capacity

theorem view_valid (v : View) (h : viewCheck v) (allocation : Allocation v) : v.Valid := by
  refine ⟨h.1,h.2.1,?_,allocation⟩
  rcases h.2.2 with hr|hc|h|h
  · exact Or.inr (by simp [hr])
  · exact Or.inl (by simp [hc])
  · exact Or.inl h
  · exact Or.inr h

theorem admitted_domain (p : Metadata) (h : checkInputs p = .ok ())
    (a : Allocation p.config.A) (b : Allocation p.config.B) (c : Allocation p.config.C) :
    Public.Admitted p.config := by
  obtain ⟨_,hc,hi⟩ := (checkInputs_iff p).1 h
  simp only [configChecks,decide_eq_true_eq] at hc
  obtain ⟨bm,bn,bk,am,ak,bk',bn',cm,cn,av,bv,cv⟩ := hc
  exact ⟨⟨(block_atoms _ bm).1,(block_atoms _ bn).1,(block_atoms _ bk).1,(block_atoms _ bk).2,
    ⟨am,ak⟩,⟨bk',bn'⟩,⟨cm,cn⟩,view_valid _ av a,view_valid _ bv b,view_valid _ cv c⟩,
    (indexChecks_iff _).1 hi⟩

#axiomsClean admitted_domain
#axiomsClean checkInputs_iff
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.MatmulKernel.Checked
