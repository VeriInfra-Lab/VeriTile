import bench.tritonbench_g.matmul_dequant_int4.Sources.Source
namespace VeriTile.Bench.Repairs.MatmulDequantInt4.Strides
set_option maxHeartbeats 4000000

structure Layout where
  row : Nat
  column : Nat
  deriving DecidableEq, Repr
def Layout.Admitted (l : Layout) (rows columns : Nat) :=
  (l.row=columns ∨ rows=1 ∧ l.row=1) ∧
  (l.column=1 ∨ columns=1 ∧ l.column=rows)
def Layout.address (l : Layout) (r c : Nat) := r*l.row+c*l.column
def canonical (columns : Nat) : Layout := ⟨columns,1⟩
/-- Transpose of an ordinary [columns,rows] allocation followed by contiguous.
Singleton transposes are already contiguous, so their original strides survive. -/
def quantReturned (rows columns : Nat) : Layout :=
  if rows=1 ∨ columns=1 then ⟨1,rows⟩ else canonical columns

theorem canonical_admitted (rows columns : Nat) : (canonical columns).Admitted rows columns :=
  ⟨Or.inl rfl,Or.inl rfl⟩
theorem returned_admitted (rows columns : Nat) : (quantReturned rows columns).Admitted rows columns := by
  by_cases hr : rows=1
  · subst rows;simp [quantReturned,Layout.Admitted]
  · by_cases hc : columns=1
    · subst columns;simp [quantReturned,Layout.Admitted]
    · simp [quantReturned,hr,hc,canonical,Layout.Admitted]
theorem active_address (l : Layout) (rows columns r c : Nat) (h : l.Admitted rows columns)
    (hr : r<rows) (hc : c<columns) : l.address r c=r*columns+c := by
  rcases h with ⟨hrow,hcol⟩
  rcases hrow with hrow|⟨hrs,hrow⟩ <;> rcases hcol with hcol|⟨hcs,hcol⟩
  · simp [Layout.address,hrow,hcol]
  · have : c=0 := by omega
    simp [Layout.address,hrow,this]
  · have : r=0 := by omega
    simp [Layout.address,hcol,this]
  · have hr0 : r=0 := by omega
    have hc0 : c=0 := by omega
    simp [Layout.address,hr0,hc0]

structure Packed where
  weight : Layout
  scale : Layout
  zero : Layout
def Packed.Admitted (p : Packed) (s : Shape) :=
  p.weight.Admitted (s.k/8) s.n ∧ p.scale.Admitted (s.k/s.group) s.n ∧
  p.zero.Admitted (s.k/s.group) (s.n/8)
def qAddress (p : Packed) (k n : Nat) := p.weight.address (k/8) n
def sAddress (p : Packed) (s : Shape) (k n : Nat) := p.scale.address (k/s.group) n
def zAddress (p : Packed) (s : Shape) (k n : Nat) := p.zero.address (k/s.group) (n/8)

theorem active_packed (p : Packed) (s : Shape) (hs : s.Valid) (h : p.Admitted s)
    (k n : Nat) (hk : k<s.k) (hn : n<s.n) :
    qAddress p k n=MatmulDequantInt4.qAddress s k n ∧
    sAddress p s k n=MatmulDequantInt4.sAddress s k n ∧
    zAddress p s k n=MatmulDequantInt4.zAddress s k n := by
  have hk8 : k/8<s.k/8 := (Nat.div_lt_iff_lt_mul (by decide : 0<8)).mpr (by simpa [Nat.div_mul_cancel hs.k8] using hk)
  have hkg : k/s.group<s.k/s.group := (Nat.div_lt_iff_lt_mul hs.gpos).mpr (by simpa [Nat.div_mul_cancel hs.groups] using hk)
  have hn8 : n/8<s.n/8 := (Nat.div_lt_iff_lt_mul (by decide : 0<8)).mpr (by simpa [Nat.div_mul_cancel hs.n8] using hn)
  exact ⟨active_address p.weight _ _ _ _ h.1 hk8 hn,
    active_address p.scale _ _ _ _ h.2.1 hkg hn,active_address p.zero _ _ _ _ h.2.2 hkg hn8⟩

/-- The candidate guard uses actual strides even for inactive pointers. Active
address equivalence alone would not justify the zero-point singleton column. -/
structure Guard (p : Packed) (s : Shape) : Prop where
  weight : (Source.paddedK s/8+1)*p.weight.row+Source.paddedN s*p.weight.column<2^31
  scale : (Source.paddedK s/s.group+1)*p.scale.row+Source.paddedN s*p.scale.column<2^31
  zero : (Source.paddedK s/s.group+1)*p.zero.row+(Source.paddedN s/8)*p.zero.column<2^31
  output : Source.paddedK s*s.n+Source.paddedN s<2^31
  grid : Source.paddedK s*Source.paddedN s<2^31
  k : s.k<2^31
  n : s.n<2^31
  group : s.group<2^31
  packed : s.k*s.n<2^31

theorem formed (p : Packed) (s : Shape) (h : Guard p s) (k n : Nat)
    (hk : k<Source.paddedK s) (hn : n<Source.paddedN s) :
    qAddress p k n<2^31 ∧ sAddress p s k n<2^31 ∧ zAddress p s k n<2^31 ∧ k*s.n+n<2^31 := by
  have kq := Nat.div_le_div_right (show k≤Source.paddedK s by omega) (c:=8)
  have kg := Nat.div_le_div_right (show k≤Source.paddedK s by omega) (c:=s.group)
  have nq := Nat.div_le_div_right (show n≤Source.paddedN s by omega) (c:=8)
  have qm := Nat.mul_le_mul_right p.weight.row kq
  have sm := Nat.mul_le_mul_right p.scale.row kg
  have zm := Nat.mul_le_mul_right p.zero.row kg
  have qn := Nat.mul_le_mul_right p.weight.column (show n≤Source.paddedN s by omega)
  have sn := Nat.mul_le_mul_right p.scale.column (show n≤Source.paddedN s by omega)
  have zn := Nat.mul_le_mul_right p.zero.column nq
  have om := Nat.mul_le_mul_right s.n (show k≤Source.paddedK s by omega)
  have hQ := h.weight;have hS := h.scale;have hZ := h.zero;have hO := h.output
  simp only [Nat.add_mul,Nat.one_mul] at hQ hS hZ
  dsimp [qAddress,sAddress,zAddress,Layout.address]
  exact ⟨by omega,by omega,by omega,by omega⟩

theorem singleton_column_example :
    (quantReturned 3 1).address 2 0=2 ∧ (quantReturned 3 1).address 2 7=23 := by decide +kernel
theorem singleton_scale_example :
    quantReturned 1 256=⟨1,1⟩ ∧ quantReturned 1 256≠canonical 256 := by decide +kernel

#axiomsClean returned_admitted
#axiomsClean active_address
#axiomsClean active_packed
#axiomsClean formed
#axiomsClean singleton_column_example
#axiomsClean singleton_scale_example
end VeriTile.Bench.Repairs.MatmulDequantInt4.Strides
