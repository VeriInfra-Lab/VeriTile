import bench.repairs.tritonbench_g.parallel_attention.Proofs.HistoricalFPEquiv

namespace VeriTile.Bench.Repairs.ParallelAttention
noncomputable section

/-- Selected public scale words for K=1,...,128. Matching Python power and
argument conversion to this table is a local scalar obligation, not a tensor
or output premise. The table is independent of Q/K/V and all cotangents. -/
def publicScaleWords : List Nat := [1065353216, 1060439283, 1058262330, 1056964608, 1055193390, 1053885932, 1052869775, 1052050675, 1051372203, 1050798235, 1050304434, 1049873722, 1049493717, 1049155191, 1048851109, 1048576000, 1048075074, 1047616495, 1047194613, 1046804782, 1046443139, 1046106442, 1045791949, 1045497324, 1045220557, 1044959915, 1044713891, 1044481167, 1044260587, 1044051130, 1043851892, 1043662067, 1043480938, 1043307859, 1043142252, 1042983595, 1042831413, 1042685279, 1042544802, 1042409627, 1042279428, 1042153907, 1042032790, 1041915826, 1041802783, 1041693446, 1041587618, 1041485114, 1041385765, 1041289411, 1041195905, 1041105109, 1041016895, 1040931144, 1040847741, 1040766583, 1040687570, 1040610610, 1040535614, 1040462501, 1040391194, 1040321618, 1040253706, 1040187392, 1040057836, 1039931236, 1039807481, 1039686466, 1039568092, 1039452263, 1039338890, 1039227887, 1039119173, 1039012670, 1038908304, 1038806005, 1038705706, 1038607341, 1038510850, 1038416174, 1038323257, 1038232045, 1038142486, 1038054531, 1037968133, 1037883246, 1037799827, 1037717834, 1037637227, 1037557967, 1037480017, 1037403341, 1037327906, 1037253678, 1037180624, 1037108716, 1037037922, 1036968214, 1036899565, 1036831949, 1036765339, 1036699711, 1036635042, 1036571307, 1036508485, 1036446554, 1036385494, 1036325283, 1036265903, 1036207334, 1036149559, 1036092559, 1036036318, 1035980818, 1035926044, 1035871979, 1035818609, 1035765919, 1035713895, 1035662522, 1035611788, 1035561678, 1035512181, 1035463284, 1035414975, 1035367242, 1035320073, 1035273459]

def softwareScale (n : Nat) (enabled : Bool) : FP32Word :=
  if enabled then VeriTile.Triton.FP.Value.ofBits .fp32 (publicScaleWords[n-1]?.getD 0x7fc00000)
  else VeriTile.Triton.FP.Value.ofBits .fp32 0x3f800000

/-- Concrete executable IEEE-value software profile. Native conformance is
still exactly the fixed-arity ScalarAgreement premise; this definition does
not assert hardware handling of NaN payloads or subnormals. Same-dtype casts
preserve bits, as the selected source contains only FP32-to-FP32 casts. -/
def softwareOps : Ops FP32Word where
  zero := VeriTile.Triton.FP.Value.ofBits .fp32 0
  one := VeriTile.Triton.FP.Value.ofBits .fp32 0x3f800000
  two := VeriTile.Triton.FP.Value.ofBits .fp32 0x40000000
  add := VeriTile.Triton.FP.add .ieeeValue
  mul := VeriTile.Triton.FP.mul .ieeeValue
  fma := VeriTile.Triton.FP.fma .ieeeValue
  div := VeriTile.Triton.FP.div .ieeeValue
  neg := VeriTile.Triton.FP.Value.neg
  cast := id
  scale := softwareScale

theorem fp32_software_words (native : Ops FP32Word)
    (primitives : ScalarAgreement native softwareOps) (c : CompilerLayout) (s : Shape) (b : Bases)
    (trees : OutputKind → SumPlan) (scratch output : OutputKind → Nat) (kinds : List OutputKind)
    (ks hs : List Event) (layout : RunLayout s b trees scratch output kinds ks hs) (m : Memory FP32Word) :
    (∀ kind,kind∈kinds → ∀ i,i < kind.cells s →
      compiledRun native c s b trees scratch output ks hs m (output kind+i)=
        nativeWordReference softwareOps c s kind (trees kind) (currentInputs softwareOps s b m) i) ∧
    (∀ address,(∀ ev∈ks,tensorAddress scratch ev ≠ address) →
      (∀ ev∈hs,tensorAddress output ev ≠ address) →
      compiledRun native c s b trees scratch output ks hs m address=m address) :=
  fp32_public_words native softwareOps primitives c s b trees scratch output kinds ks hs layout m

#axiomsClean fp32_software_words
#specNonCircular softwareOps avoiding [compiledRun,nativeWordReference]
end
end VeriTile.Bench.Repairs.ParallelAttention
