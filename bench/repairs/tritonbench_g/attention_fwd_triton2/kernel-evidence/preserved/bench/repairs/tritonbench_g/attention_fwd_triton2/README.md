# Checked BF16-returning attention

The supported caller uses ordinary strided CUDA FP32 Q/K/V and scales on one device, resolved views, Q/K/V shape `[Z,H,N,96]`, canonical dense input strides, and packed scales `[Z,H,ceil(N/128)]` and `[Z,H,ceil(N/64)]`. Read-only input aliases are allowed. No copy or padding is added. Empty Z/H/N returns the fresh BF16 output and preserves the source's unused FP32 M allocation without launching; its stride checks are bypassed exactly as in the caller.

The actual `_launch_envelope` and `_check_pointer` guards bound all formed coordinates, including masked tails and final pointer advances, and enforce natural alignment and absolute unsigned64 bounds. `Contract.admittedDomain` derives the accepted machine domain from these checks and the ordinary fresh-output stride fact. `Contract.admittedABI` derives checked metadata from raw descriptors. Dense finite access bounds and output injectivity remain proved by the accepted `Machine.storage`, not supplied as access premises.

`Correct.lean` exports `Checked.checked_real`, which proves the independent all-key base-2 normalized attention formula, finite returned BF16-tagged values, read-only input/frame, and machine closure. `FPEquiv.lean` exports `Checked.checked_native`, which preserves the whole source graph: FP32 QK with default TF32, FP16 P and V, FP16 PV dot result widened into FP32 accumulation, source maximum/denominator updates, and final BF16 narrowing. It includes physical output owners and reduction order. The separate numerical transport endpoint defers only the accepted literal/scalar/conversion/fixed-ISA families. Compiler, allocation, and runtime interpretation remain ordinary boundaries; there is no native-equals-Real claim.

Validation includes the accepted 27-module closure and four checked modules, plus executable host checks for tails, aliases, empty strides, packed scales, launch limits, and actual pointer rejection. Tests use metadata/allocation/launch spies, not GPU execution.

### Launch-device admission correction (2026-10-19)

Prior candidate SHA-256: `eedab60247bb8128eb6bfe357474f1f18d8a3ac9a55a92e6d8a6769c739af322`.
The installed Triton JIT chooses the current CUDA device and stream; equal
input devices alone did not select the owning tensor device. The actual source
submission now enters `torch.cuda.device(q.device)` and restores the
ambient caller device. Formal host device/stream observations refer to this
effective context. CPU controls check a cuda:1 launch from an ambient cuda:0
caller and restoration on success and submission exception. Kernel AST and
numerical semantics are unchanged. The prior hash is retained as acceptance
correction evidence; no native GPU/runtime conformance is claimed.
