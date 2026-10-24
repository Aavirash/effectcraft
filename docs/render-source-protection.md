# Render source protection

The render queue refuses an output that would replace footage, an attached proxy, or the
project file. It checks every output module and segment before dispatching the item. A refused
item is marked Failed with its reason; other valid queued items can still render. Render logs
are checked too. No render parameter overrides this protection.

Checks compare normalized paths and native file identities, so an existing symbolic link,
hard link, junction or Windows short name cannot disguise a source as another output file.
Source identities are indexed once per check and matching identities are verified. The index
does not keep file handles open for the lifetime of a long sequence.

Default output names move aside when they would replace a protected file. This also applies to
Create Proxy: recreating a proxy chooses a new default name and attaches the new proxy, leaving
the previous proxy file intact. An explicit output path is kept as entered and refused if it
would replace a protected file.

Image-sequence checks inspect existing frames in the requested range. Skip Existing Files leaves
existing frames alone. If a frame number appears in a folder name, the checker probes each
frame separately and refuses ranges over 100,000 frames instead of silently checking a prefix.
Choose a fixed output folder or a shorter range for those exports.

This guard does not introduce an overwrite flag, confirmation dialog, or changed replacement
defaults for the CLI, scripts or MCP. Replacement consent and staging ordinary output files
belong to the exporter. The source check runs before export; filesystem changes between that
check and a write still require exporter-side revalidation.
