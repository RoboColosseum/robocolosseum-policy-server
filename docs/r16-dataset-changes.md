# R16 action evidence in the existing policy runtime

This PR pairs with the client data-recording migration. Keep both PRs unmerged until repository review and isolated Linux/hardware A/B acceptance are complete.

## Scope and architecture

The existing local and remote runtimes carry original model actions separately from executable actions. ActionPlan adds fields 8-11; existing fields, including done, retain their contract. Legacy readers can ignore evidence. Missing evidence is not fabricated from clipped actions.

YAM adapters retain output before safety clipping. Dream records complete replies and interpolation provenance, including unused tails; terminal drain waits for an already requested reply without additional inference. Auto forwards evidence while retaining CaP-X and observation extras.

The remote runner and SDK accept actions/evidence mappings while preserving session identity, deadline, duplicate-sequence, dimensions, horizon and done checks. Invalid optional evidence is marked unavailable without replacing valid executable commands. PNG is additive; raw RGB/JPEG remain supported.

Use existing commands, backend plugins and extras. No service, configuration group or dependency is added. The paired client carries the pinned I2RT patch; this repository does not apply a driver patch. The first-command tolerance change is excluded from the client.

## Validation and remaining gates

Paired descriptors and lossless RGB/PNG round trips pass. Fake Router tests preserve raw float64 blocks separately from float32 commands and verify invalid evidence leaves commands intact. SDK/reconnect tests pass. Dream execution equivalence uses an immutable upstream fixture. The wheel builds.

Windows regression: 320 passed, 10 skipped, one original POSIX process-launch/SIGINT failure (test_policy_process_stops_on_sigint). Linux validation and actual paired A/B recording remain required. Old r16 field recordings do not certify this migration.

## Rollback

Keep previous paired environments/configuration and select them together. Review or installation does not authorize production restart, motion, inference or data upload.
