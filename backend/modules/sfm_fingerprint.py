"""Staleness guard for the SfM (sparse) workspace that COLMAP MVS reuses.

The MVS page can reuse an existing ``SFM_output/dense`` workspace (the ``reuse``
step) and run only patch-match + fusion on it. That is a big time saver, but it
used to key off nothing more than "the directory exists" -- change a SfM
parameter, click COLMAP MVS, and the old workspace (including a frozen
``stereo/patch-match.cfg`` and a sparse model built under the old settings) was
copied into ``MVS_output/colmap`` without a word.

The guard works by *fingerprinting* the SfM settings that a run actually
applies, and stamping that fingerprint into the workspace it produces:

    SFM_output/dense/.sfm_stamp.json

``step_undistort`` in sfm_cli.py writes the stamp (it is the step that produces
the workspace), removing any previous one first so an interrupted run leaves no
stamp rather than a wrong one. Before reusing a workspace the backend compares
the stamped fingerprint against the settings it would use now and refuses on a
mismatch -- or on a missing stamp, which is the normal state for projects
produced before this guard existed.

Only this module computes the fingerprint; the CLI just relocates the opaque
blob it is handed. One implementation means the two sides cannot drift.
"""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from modules.logger import get_logger
from modules.models import COLMAP_SFM_CLI_FLAGS, ColmapSFMParams

logger = get_logger(__name__)

#: Bump only when the field whitelist, a normalisation rule or the meaning of an
#: option changes. Adding a new optional parameter does NOT need a bump: unset
#: values are omitted from the payload, so old stamps stay valid.
SFM_FINGERPRINT_SCHEMA_VERSION = 1

#: Written by step_undistort into the dense workspace it just produced.
STAMP_FILENAME = ".sfm_stamp.json"

#: Settings that live in the project's ``config`` (rather than in
#: ``ColmapSFMParams``) but still decide what the sparse model looks like.
#: An explicit whitelist on purpose -- dumping the whole config would drag in
#: ``steps`` / ``use_gpu`` / ``gpu_index`` / ``no_geom_consistency`` /
#: ``convert_output_suffix``, none of which change the reconstruction.
#:
#: ``use_gpu`` and ``gpu_index`` are deliberately excluded: they are resource
#: switches, not quality knobs. Including them would invalidate a perfectly good
#: workspace when the user switches machines or debugs on CPU. The guard answers
#: "did the user ask for a different reconstruction?", not "is this bit-for-bit
#: reproducible?".
CONFIG_FINGERPRINT_FIELDS = (
    "camera_model",
    "single_camera",
    "matcher",
    "vocab_tree_path",
    "reconstruction_name",
    "max_num_features",
)

#: Defaults applied when a config key is absent, mirroring build_sfm_command /
#: _start_pipeline. Resolving them here means a task.json that gained an explicit
#: key equal to the default does not look like a parameter change.
_CONFIG_DEFAULTS: dict[str, Any] = {
    "camera_model": "SIMPLE_RADIAL",
    "single_camera": True,
    "matcher": "exhaustive",
    "reconstruction_name": "0",
    "max_num_features": 0,
}


def _normalize(value: Any) -> Any:
    """Coerce to a JSON primitive so equal settings always hash equally.

    bools become 0/1 (config stores ``True`` where the form sends ``1``),
    Paths become str, and enums unwrap to their value.
    """
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return _normalize(value.value)
    return value


def effective_sfm_config(
    config: Mapping[str, Any] | None,
    sfm_params: ColmapSFMParams | None,
    *,
    undistort_max_image_size: int | None = None,
    num_patch_match_src_images: int | None = None,
) -> dict[str, Any]:
    """Flatten the SfM settings a run would apply into one canonical mapping.

    Only entries that actually change behaviour are kept: unset (None) options
    and the ``max_num_features=0`` "use COLMAP's default" sentinel are dropped,
    and a dormant ``vocab_tree_path`` is dropped unless the matcher is
    ``vocab_tree`` (otherwise editing an unused path would report a stale
    workspace). The result is both the hashed payload and the ``params`` block
    stored in the stamp, so the change list a user sees is derived from exactly
    the values that were compared.
    """
    config = config or {}
    effective: dict[str, Any] = {}

    def resolve(name: str) -> Any:
        value = config.get(name)
        if value is None:
            value = _CONFIG_DEFAULTS.get(name)
        return _normalize(value)

    for name in CONFIG_FINGERPRINT_FIELDS:
        value = resolve(name)
        if name == "max_num_features":
            # The CLI only forwards this flag when it is > 0; 0 means "let
            # COLMAP use its default", which is the same thing as not setting it.
            if not value:
                continue
        elif name == "vocab_tree_path":
            # Only meaningful for the vocabulary-tree matcher: the path is baked
            # into the database during matching, so no other matcher reads it.
            # An empty string counts as "not configured" and is dropped too.
            if resolve("matcher") != "vocab_tree" or not value:
                continue
        effective[name] = value

    # Everything the user can override on the SfM page. Unset fields are omitted,
    # which is what keeps the payload stable when a new optional knob is added.
    if sfm_params is not None:
        for field in COLMAP_SFM_CLI_FLAGS:
            value = _normalize(getattr(sfm_params, field))
            if value is not None:
                effective[field] = value

    # These two are what image_undistorter bakes into the dense workspace (they
    # shape dense/images and the frozen stereo/patch-match.cfg). The backend does
    # not pass them today, so they resolve to None and drop out -- they are
    # covered here so that wiring them up later cannot silently bypass the guard.
    for name, value in (
        ("undistort_max_image_size", undistort_max_image_size),
        ("num_patch_match_src_images", num_patch_match_src_images),
    ):
        value = _normalize(value)
        if value is not None:
            effective[name] = value

    return effective


def compute_sfm_fingerprint(effective: Mapping[str, Any]) -> str:
    """Hash a canonicalised mapping into a stable hex digest.

    The schema version is part of the payload, so bumping it invalidates every
    existing stamp. Keys are sorted and separators are tight to make the digest
    independent of dict ordering.
    """
    payload = {"schema_version": SFM_FINGERPRINT_SCHEMA_VERSION, **effective}
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def build_stamp_json(effective: Mapping[str, Any]) -> str:
    """Render the opaque blob the CLI writes verbatim next to the workspace.

    Carries the full effective mapping alongside the digest so a mismatch can
    name the parameters that changed instead of just saying "stale".
    """
    return json.dumps(
        {
            "schema_version": SFM_FINGERPRINT_SCHEMA_VERSION,
            "fingerprint": compute_sfm_fingerprint(effective),
            "params": dict(effective),
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def read_sfm_stamp(dense_dir: Path | str) -> dict[str, Any] | None:
    """Read the stamp next to a dense workspace, or None if unusable.

    A missing stamp is the normal state for projects built before this guard, and
    an unreadable one means someone hand-edited it. Both return None so callers
    treat them the same way: refuse to reuse, and say why.
    """
    path = Path(dense_dir) / STAMP_FILENAME
    try:
        with open(path, encoding="utf-8") as fh:
            stamp = json.load(fh)
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Ignoring unreadable SfM stamp %s: %s", path, exc)
        return None
    if not isinstance(stamp, dict):
        logger.warning("Ignoring malformed SfM stamp %s (not an object)", path)
        return None
    return stamp


def diff_sfm_params(
    produced: Mapping[str, Any], current: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """List the settings that differ, for the message shown to the user.

    Absent keys read as None: a parameter that was set when the workspace was
    built and is now back at its default is a change worth reporting.
    """
    return [
        {"name": name, "produced": produced.get(name), "current": current.get(name)}
        for name in sorted(set(produced) | set(current))
        if produced.get(name) != current.get(name)
    ]
