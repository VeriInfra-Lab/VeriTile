# Explicit gzip JSON storage support

The shared JSON helper now supports explicit `.json.gz` paths while retaining
plain JSON behavior. Full-stream gzip validation, duplicate-key rejection,
canonical object hashing and raw-byte receipt binding remain enforced.

The root reviewed the production diff and tests, verified all203 frozen producer
artifacts, and ran228 installed-source tests successfully. The producer's actual
plain/gzip scope comparison retained one verified plus three scopes with deferred
numerics and179 pending. This storage change adds no accepted program.

`producer-evidence.zip` losslessly retains every producer-receipt artifact and
the receipt itself. `archive-index.json` binds every uncompressed member hash
and the actual archive bytes. Initial failed attempts remain included. The
snapshots are evidence only; no active ledger or historical file is migrated.
Test comparator stubs do not establish official comparator or GPU correctness.
