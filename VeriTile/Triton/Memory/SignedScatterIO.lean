import VeriTile.Triton.Memory.KernelSpec.Scatter
import VeriTile.Meta.StatementAudit

/-! Gather/scatter IO with nonnegative mathematical metadata encoded in actual
signed integer cells. This keeps the flat allocation, source safety, successful
execution, injective active readback and unconditional exterior frame. -/
namespace VeriTile.Triton

structure SignedScatterIO where
  /-- The kernel being specified. -/
  kernel : ComputeKernel
  /-- The contract executes only a successfully projected kernel. The default
  discharges transparent supported kernels; abstract kernels need a witness. -/
  projection : kernel.toAlgorithm? = Except.ok kernel.toAlgKernel := by
    first | rfl | simp [ComputeKernel.toAlgKernel]
  /-- First float scalar slot's buffer (one ℝ cell per program). -/
  fbuf1 : RegionName
  /-- Second float scalar slot's buffer. -/
  fbuf2 : RegionName
  /-- Third float scalar slot's buffer. -/
  fbuf3 : RegionName
  /-- The nonnegative signed-integer metadata buffer carrying both scalar slots (the
  consumer's cumulative-sequence-length array). -/
  mbuf : RegionName
  /-- Index buffer (nonnegative signed-integer tile — the gather/scatter token ids). -/
  idbuf : RegionName
  /-- Payload buffer (nonnegative signed-integer tile — the per-token counts). -/
  cntbuf : RegionName
  /-- Float data buffer, gather-read at the index-dependent window. -/
  inp : RegionName
  /-- Output buffer, scatter-written (the in-place consumer sets
  `out := inp`). -/
  out : RegionName
  /-- Tile length: each program instance owns `B`-lane windows. -/
  B : Nat
  /-- First float slot's cell address for program `(pid₀, pid₁)`. -/
  fwin1 : Nat → Nat → Nat
  /-- Second float slot's cell address. -/
  fwin2 : Nat → Nat → Nat
  /-- Third float slot's cell address. -/
  fwin3 : Nat → Nat → Nat
  /-- First nonnegative signed-integer slot's cell address in `mbuf`. -/
  mwin1 : Nat → Nat → Nat
  /-- Second nonnegative signed-integer slot's cell address in `mbuf`. -/
  mwin2 : Nat → Nat → Nat
  /-- Lane `j`'s `idbuf` read address at `(pid₀, pid₁, m₁, m₂)` — the
  loaded nonnegative signed-integer scalars are ordinary named arguments. -/
  readi : Nat → Nat → Nat → Nat → Fin B → Nat
  /-- Lane `j`'s `cntbuf` read address. -/
  readc : Nat → Nat → Nat → Nat → Fin B → Nat
  /-- Lane `j`'s `inp` **gather** read address, given the loaded index
  tile. -/
  read : Nat → Nat → Nat → Nat → (Fin B → Nat) → Fin B → Nat
  /-- Read-active lanes, given the loaded nonnegative signed-integer scalars (the consumer's
  `m₁ + j < m₂` batch window). -/
  mask : Nat → Nat → Nat → Nat → Fin B → Prop
  /-- Lane `j`'s **scatter** write address, given the loaded index tile. -/
  write : Nat → Nat → Nat → Nat → (Fin B → Nat) → Fin B → Nat
  /-- Write-active lanes given the loaded index tile; defaults to `mask`. -/
  writeMask : Nat → Nat → Nat → Nat → (Fin B → Nat) → Fin B → Prop :=
    fun p₀ p₁ m₁ m₂ _ j => mask p₀ p₁ m₁ m₂ j

namespace SignedScatterIO

/-- No two write-active lanes share a scatter destination — the
per-pinned-context injectivity that scatter readback needs. For
`apply_penalty` this is the distinct-active-token-ids side condition
(`hUniq`). -/
def WriteInj (io : SignedScatterIO) (p₀ p₁ m₁ m₂ : Nat)
    (ids : Fin io.B → Nat) : Prop :=
  ∀ j k : Fin io.B, io.writeMask p₀ p₁ m₁ m₂ ids j →
    io.writeMask p₀ p₁ m₁ m₂ ids k →
    io.write p₀ p₁ m₁ m₂ ids j = io.write p₀ p₁ m₁ m₂ ids k → j = k

/-- `io.Implements f` — the penalty gather–scatter masked Hoare triple.
The float slots `g₁ g₂ g₃`, the nonnegative signed-integer slots `m₁ m₂`, the index tile
`ids`, the payload tile `cnts`, and the gathered data tile `xs` are all
universally quantified and pinned inside the memory precondition, so the
window/mask/value contracts speak about the *loaded* values throughout.
The readback leg is guarded by `WriteInj` (see the struct docstring);
the frame excludes the raw (ungated) scatter cells. -/
def Implements (io : SignedScatterIO)
    (f : Nat → Nat → ℝ → ℝ → ℝ → Nat → Nat → (Fin io.B → Nat) →
      (Fin io.B → Nat) → (Fin io.B → ℝ) → Fin io.B → ℝ) : Prop :=
  ∀ A : FlatAlloc,
    A.Disjoint →
    A.regions = [io.fbuf1, io.fbuf2, io.fbuf3, io.mbuf, io.idbuf,
      io.cntbuf, io.inp, io.out] →
    (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ pid₀ pid₁ : Nat,
  ∀ (g₁ g₂ g₃ : ℝ) (m₁ m₂ : Nat) (ids cnts : Fin io.B → Nat)
    (xs : Fin io.B → ℝ) (s₀ : BlockState),
    s₀.pids 0 = pid₀ →
    s₀.pids 1 = pid₁ →
    s₀.undef = (fun _ _ => 0) →
    io.fwin1 pid₀ pid₁ < A.extent io.fbuf1 →
    io.fwin2 pid₀ pid₁ < A.extent io.fbuf2 →
    io.fwin3 pid₀ pid₁ < A.extent io.fbuf3 →
    io.mwin1 pid₀ pid₁ < A.extent io.mbuf →
    io.mwin2 pid₀ pid₁ < A.extent io.mbuf →
    (∀ j : Fin io.B, io.mask pid₀ pid₁ m₁ m₂ j →
      io.readi pid₀ pid₁ m₁ m₂ j < A.extent io.idbuf) →
    (∀ j : Fin io.B, io.mask pid₀ pid₁ m₁ m₂ j →
      io.readc pid₀ pid₁ m₁ m₂ j < A.extent io.cntbuf) →
    (∀ j : Fin io.B, io.mask pid₀ pid₁ m₁ m₂ j →
      io.read pid₀ pid₁ m₁ m₂ ids j < A.extent io.inp) →
    (∀ j : Fin io.B, io.writeMask pid₀ pid₁ m₁ m₂ ids j →
      io.write pid₀ pid₁ m₁ m₂ ids j < A.extent io.out) →
    s₀.readMem io.fbuf1 (io.fwin1 pid₀ pid₁) = g₁ →
    s₀.readMem io.fbuf2 (io.fwin2 pid₀ pid₁) = g₂ →
    s₀.readMem io.fbuf3 (io.fwin3 pid₀ pid₁) = g₃ →
    s₀.readMemValue .int io.mbuf (io.mwin1 pid₀ pid₁) = (m₁ : Int) →
    s₀.readMemValue .int io.mbuf (io.mwin2 pid₀ pid₁) = (m₂ : Int) →
    (∀ j : Fin io.B, io.mask pid₀ pid₁ m₁ m₂ j →
      s₀.readMemValue .int io.idbuf (io.readi pid₀ pid₁ m₁ m₂ j) = (ids j : Int)) →
    (∀ j : Fin io.B, io.mask pid₀ pid₁ m₁ m₂ j →
      s₀.readMemValue .int io.cntbuf (io.readc pid₀ pid₁ m₁ m₂ j)
        = (cnts j : Int)) →
    (∀ j : Fin io.B, io.mask pid₀ pid₁ m₁ m₂ j →
      s₀.readMem io.inp (io.read pid₀ pid₁ m₁ m₂ ids j) = xs j) →
    ∃ s',
      exec (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s₀)
        = some s'
      ∧ (io.WriteInj pid₀ pid₁ m₁ m₂ ids →
          ∀ j : Fin io.B, io.writeMask pid₀ pid₁ m₁ m₂ ids j →
            s'.readMem A.flat
                (A.addr io.out (io.write pid₀ pid₁ m₁ m₂ ids j))
              = f pid₀ pid₁ g₁ g₂ g₃ m₁ m₂ ids cnts xs j)
      ∧ (∀ r' o',
          (r' ≠ A.flat ∨
            (∀ j : Fin io.B, io.writeMask pid₀ pid₁ m₁ m₂ ids j →
              o' ≠ A.addr io.out (io.write pid₀ pid₁ m₁ m₂ ids j))) →
          s'.mem r' o' = (A.flattenState s₀).mem r' o')

@[inherit_doc] scoped infix:25 " ⊨ " => SignedScatterIO.Implements

theorem Implements.intro (io : SignedScatterIO)
    {f : Nat → Nat → ℝ → ℝ → ℝ → Nat → Nat → (Fin io.B → Nat) →
      (Fin io.B → Nat) → (Fin io.B → ℝ) → Fin io.B → ℝ}
    (hok : (io.kernel.toAlgKernel).FlattenOk)
    (hts : ∀ (bounds : RegionBounds) (s : BlockState) (m₁ m₂ : Nat)
        (ids : Fin io.B → Nat),
      s.readMemValue .int io.mbuf (io.mwin1 (s.pids 0) (s.pids 1)) = (m₁ : Int) →
      s.readMemValue .int io.mbuf (io.mwin2 (s.pids 0) (s.pids 1)) = (m₂ : Int) →
      (∀ j : Fin io.B, io.mask (s.pids 0) (s.pids 1) m₁ m₂ j →
        s.readMemValue .int io.idbuf
          (io.readi (s.pids 0) (s.pids 1) m₁ m₂ j) = (ids j : Int)) →
      io.fwin1 (s.pids 0) (s.pids 1) < bounds io.fbuf1 →
      io.fwin2 (s.pids 0) (s.pids 1) < bounds io.fbuf2 →
      io.fwin3 (s.pids 0) (s.pids 1) < bounds io.fbuf3 →
      io.mwin1 (s.pids 0) (s.pids 1) < bounds io.mbuf →
      io.mwin2 (s.pids 0) (s.pids 1) < bounds io.mbuf →
      (∀ j : Fin io.B, io.mask (s.pids 0) (s.pids 1) m₁ m₂ j →
        io.readi (s.pids 0) (s.pids 1) m₁ m₂ j < bounds io.idbuf) →
      (∀ j : Fin io.B, io.mask (s.pids 0) (s.pids 1) m₁ m₂ j →
        io.readc (s.pids 0) (s.pids 1) m₁ m₂ j < bounds io.cntbuf) →
      (∀ j : Fin io.B, io.mask (s.pids 0) (s.pids 1) m₁ m₂ j →
        io.read (s.pids 0) (s.pids 1) m₁ m₂ ids j < bounds io.inp) →
      (∀ j : Fin io.B, io.writeMask (s.pids 0) (s.pids 1) m₁ m₂ ids j →
        io.write (s.pids 0) (s.pids 1) m₁ m₂ ids j < bounds io.out) →
      Kernel.TraceSafe bounds (io.kernel.toAlgKernel) s)
    (hrun : ∀ (s₀ : BlockState) (g₁ g₂ g₃ : ℝ) (m₁ m₂ : Nat)
        (ids cnts : Fin io.B → Nat) (xs : Fin io.B → ℝ),
      s₀.readMem io.fbuf1 (io.fwin1 (s₀.pids 0) (s₀.pids 1)) = g₁ →
      s₀.readMem io.fbuf2 (io.fwin2 (s₀.pids 0) (s₀.pids 1)) = g₂ →
      s₀.readMem io.fbuf3 (io.fwin3 (s₀.pids 0) (s₀.pids 1)) = g₃ →
      s₀.readMemValue .int io.mbuf (io.mwin1 (s₀.pids 0) (s₀.pids 1)) = (m₁ : Int) →
      s₀.readMemValue .int io.mbuf (io.mwin2 (s₀.pids 0) (s₀.pids 1)) = (m₂ : Int) →
      (∀ j : Fin io.B, io.mask (s₀.pids 0) (s₀.pids 1) m₁ m₂ j →
        s₀.readMemValue .int io.idbuf
          (io.readi (s₀.pids 0) (s₀.pids 1) m₁ m₂ j) = (ids j : Int)) →
      (∀ j : Fin io.B, io.mask (s₀.pids 0) (s₀.pids 1) m₁ m₂ j →
        s₀.readMemValue .int io.cntbuf
          (io.readc (s₀.pids 0) (s₀.pids 1) m₁ m₂ j) = (cnts j : Int)) →
      (∀ j : Fin io.B, io.mask (s₀.pids 0) (s₀.pids 1) m₁ m₂ j →
        s₀.readMem io.inp
          (io.read (s₀.pids 0) (s₀.pids 1) m₁ m₂ ids j) = xs j) →
      ∃ s1, exec (io.kernel.toAlgKernel) s₀ = some s1
        ∧ (io.WriteInj (s₀.pids 0) (s₀.pids 1) m₁ m₂ ids →
            ∀ j : Fin io.B,
              io.writeMask (s₀.pids 0) (s₀.pids 1) m₁ m₂ ids j →
              s1.readMem io.out
                  (io.write (s₀.pids 0) (s₀.pids 1) m₁ m₂ ids j)
                = f (s₀.pids 0) (s₀.pids 1) g₁ g₂ g₃ m₁ m₂ ids cnts xs j)
        ∧ (∀ r o,
            (r ≠ io.out ∨
              ∀ j : Fin io.B,
                io.writeMask (s₀.pids 0) (s₀.pids 1) m₁ m₂ ids j →
                o ≠ io.write (s₀.pids 0) (s₀.pids 1) m₁ m₂ ids j) →
            s1.mem r o = s₀.mem r o)) :
    io.Implements f := by
  intro A hd hregs hcov p q g₁ g₂ g₃ m₁ m₂ ids cnts xs s hp hq hu
    hb₁ hb₂ hb₃ hm₁ hm₂ hi hc hx hw hg₁ hg₂ hg₃ hs he hid hcnt hxs
  subst p
  subst q
  obtain ⟨t,ht,hv,hf⟩ := hrun s g₁ g₂ g₃ m₁ m₂ ids cnts xs hg₁ hg₂ hg₃ hs he hid hcnt hxs
  have safe := hts A.extent s m₁ m₂ ids hs he hid hb₁ hb₂ hb₃ hm₁ hm₂ hi hc hx hw
  have bridge := A.exec_flatten hd hcov _ s safe hok hu
  refine ⟨A.flattenState t,?_,?_,?_⟩
  · rw [bridge,ht,Option.map_some]
  · intro inj j hj
    have hr : io.out ∈ A.regions := by rw [hregs]; simp
    have flat := A.flattenState_readMemAs hd t hr (hw j hj) .real
    simp only [BlockState.readMemAs_real] at flat
    exact (Option.some.inj flat).trans (hv inj j hj)
  · intro r' o' outside
    by_cases hr : r'=A.flat
    · subst r'
      simp only [FlatAlloc.flattenState]
      unfold FlatAlloc.readFlat
      cases hdec : A.decode o' with
      | none => rfl
      | some pair =>
        obtain ⟨r,o⟩ := pair
        obtain ⟨_,hoeq,_⟩ := A.decode_sound hdec
        change A.trCell (t.mem r o) = A.trCell (s.mem r o)
        apply congrArg A.trCell
        apply hf r o
        by_cases same : r=io.out
        · subst r
          right
          intro j hj eq
          rcases outside with impossible | miss
          · exact impossible rfl
          · apply miss j hj
            rw [hoeq,eq]
        · exact Or.inl same
    · simp only [FlatAlloc.flattenState,if_neg hr]

#axiomsClean Implements.intro
end SignedScatterIO
end VeriTile.Triton
