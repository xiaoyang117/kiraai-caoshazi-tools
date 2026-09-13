"""
Intimacy Progress Plugin for KiraAI.

Gives the LLM an explicit, always-current notion of "how far along we are" in an
intimate scene, so the model advances the narrative at a believable pace instead
of looping in place or jumping straight to the ending.

Design notes
------------
The plugin keeps a per-session state machine (a list of stages) plus an optional
desire/arousal meter, and injects a compact status block into the LLM system
prompt on every request. The LLM is also given tools to advance / retreat /
reset, so it remains the one driving the story (per the KiraAI plugin
philosophy: never hardcode the output, let the AI perceive and act).
"""

import asyncio
import json
import os
import re
import shutil
import time

from core.plugin import (
    BasePlugin, logger, on, register, Priority,
    PluginPage, PageMenu,
)
from core.chat.message_utils import KiraMessageBatchEvent
from core.chat.message_elements import Text, Image
from core.provider import LLMRequest
from core.prompt_manager import Prompt

PLUGIN_ID = "intimacy_progress"
MANIFEST_NAME = "manifest.json"

# Keys of schema.json, grouped by how the WebUI should render them. Keeping the
# order explicit means the sidebar form reads top-to-bottom like the docs do.
# Every feature has its own switch and ships off; the groups are ordered so the
# master switch and the common knobs come first.
CONFIG_SECTIONS = [
    ("basic", "基础", ["enabled", "ntr", "user_whitelist", "log_level"], True),
    ("scene", "场景识别", ["use_detect", "detect_keywords", "exit_keywords"], False),
    ("inject", "进度注入", ["inject_progress", "inject_stage_hint",
                            "force_report", "announce_tools"], False),
    ("pacing", "节奏控制", [
        "auto_advance", "auto_advance_rounds", "scene_timeout_minutes",
        "track_desire", "allow_llm_desire", "allow_llm_reset", "proactive_end",
    ], False),
    ("image", "场景配图", ["scene_image", "image_max_per_round"], True),
    ("parts", "分部位统计", ["track_parts", "part_names"], True),
    ("stages", "自定义阶段", ["stage_names"], True),
]

_MAX_BACKUPS = 5

# --------------------------------------------------------------------------- #
# Default stage definitions
# --------------------------------------------------------------------------- #

DEFAULT_STAGES = ["试探", "升温", "前戏", "结合", "高潮", "余韵"]

# Per-stage writing guidance, injected when `inject_stage_hint` is on.
# Kept intentionally short so it steers rhythm without eating much context.
DEFAULT_STAGE_HINTS = [
    "还很克制。以试探、调情、暧昧的气氛为主，重点在心理博弈与眼神/语言的拉锯，不要发生实质性接触。",
    "情绪开始升温。可以出现拥抱、亲吻、耳语等亲密接触，但节奏要慢，保留余地，不要急于进入下一步。",
    "进入前戏阶段。可以有爱抚、褪衣、逐步的身体接触。描写要细腻，注意呼吸、体温、声音等感官细节，继续吊着节奏。",
    "正式结合。这是最需要节奏控制的阶段，不要一上来就推向顶点；轮次之间要有变化，避免重复同一套动作描写。",
    "接近顶点。情绪与感官推向高峰，可以出现失控感、战栗、意识模糊等描写。不宜长时间停留在此阶段。",
    "事后余韵。以温存、喘息、依赖感、余味为主，节奏舒缓下来。此时应自然收束场景。",
]

DEFAULT_DETECT_KEYWORDS = [
    "亲吻", "亲我", "吻", "舌吻", "抚摸", "摸", "揉", "抱住", "抱我", "搂",
    "脱", "褪下", "解开", "压在", "跨坐", "坐在你身", "舔", "咬",
    "进去", "插入", "进入我", "顶", "抽", "湿了", "硬了", "想要你",
    "做爱", "上床", "睡了", "爱爱", "造爱", "给我", "要我", "开始吧",
    "骚", "淫", "高潮", "射", "choke", "hands on", "kiss me", "touch me",
]

DEFAULT_EXIT_KEYWORDS = [
    "停下", "不要了", "别弄", "住手", "结束", "算了", "累了", "歇会", "休息",
    "改天", "下次", "我走了", "走开", "滚", "stop", "don't", "enough",
    "不想要", "不想了", "别碰",
]

# Who the plugin runs for.
#
# `ntr` (default off) means "everyone": the plugin answers to any user and the
# whitelist is ignored. Switching it off narrows the plugin to the users listed
# in `user_whitelist`, which is the safe default for a shared bot.
#
# A whitelist entry matches if it equals ANY of the identifiers we can see on
# an incoming message: the sender's user_id, the session_id (group id in a
# group chat, user id in a DM), or the session title (nickname / group name).
# Matching several identities keeps the list usable without forcing the user to
# know which form to paste in.

# How much the desire meter moves per event.
DESIRE_ON_BEGIN = 8
DESIRE_ON_ADVANCE = 15
DESIRE_ON_RETREAT = 12
DESIRE_PER_ROUND = 3

# Models occasionally express a tool call as literal markup instead of a real
# function call. Because '<' must be escaped inside <text>, what reaches the
# chat is a string like `&lt;report progress="..." /&gt;` (or the unescaped
# form if the model nests it somewhere else). That leaks to the user as junk,
# so we sweep it out of outgoing text. Covers this plugin's own tool names.
_LEAKED_TOOL_MARKUP_RE = re.compile(
    r"&lt;\s*/?\s*(?:int_)?(?:report|advance|retreat|status|part|desire|reset)\b"
    r"[^&]*?(?:/&gt;|&gt;)"
    r"|<\s*/?\s*(?:int_)?(?:report|advance|retreat|status|part|desire|reset)\b"
    r"[^<>]*?(?:/?>)",
    re.IGNORECASE,
)

# Body-part tracker. The AI reports hits through the int_part tool; the WebUI
# shows a per-session tally. `part_names` in the config overrides this list.
DEFAULT_PARTS = [
    "唇", "舌", "颈", "耳", "胸", "腰", "背", "臀", "腿", "私处",
]


class IntimacyState:
    """Per-session intimacy state.

    Note: ``__slots__`` blocks weakrefs, so ``defaultdict(IntimacyState)`` would
    not work either — a plain dict plus ``get_or_create`` is used instead.
    """

    __slots__ = ("sid", "active", "stage", "rounds", "total_rounds",
                 "desire", "last_active", "stall_rounds", "started_at",
                 "parts", "part_total", "reported",
                 "scene_parts", "scene_part_total", "images_this_round")

    def __init__(self, sid: str):
        self.sid = sid
        self.active = False
        self.stage = 0
        self.rounds = 0            # rounds spent in the current stage
        self.total_rounds = 0      # rounds since the scene began
        self.desire = 0
        self.last_active = 0.0
        self.stall_rounds = 0      # consecutive rounds without an explicit move
        self.started_at = 0.0
        # Whether the model has confirmed this round's progress via `int_report`.
        # Armed on entry and re-armed after every stage change, so the model has
        # to acknowledge each step instead of silently ignoring the injected block.
        self.reported = False
        # Body-part tally for the current scene: {"唇": 3, ...}. Reset with the
        # scene, since the point is to describe *this* encounter.
        self.parts: dict[str, int] = {}
        self.part_total = 0
        # Counters for the *current* scene only. Kept apart from the lifetime
        # tally above so a late report does not inflate the scene subtotal.
        self.scene_parts: dict[str, int] = {}
        self.scene_part_total = 0
        # Images the model has already emitted this user turn. Reset on every
        # new user message; used to enforce `image_max_per_round`.
        self.images_this_round = 0

    def reset(self, keep_parts: bool = True):
        """End the current scene.

        The part tally is a *profile* of the person across encounters, so by
        default it survives a reset — only the per-scene `scene_parts` counter
        is cleared. Pass ``keep_parts=False`` to wipe the tally too.
        """
        self.active = False
        self.stage = 0
        self.rounds = 0
        self.total_rounds = 0
        self.desire = 0
        self.stall_rounds = 0
        self.started_at = 0.0
        self.scene_parts = {}
        self.scene_part_total = 0
        self.images_this_round = 0
        if not keep_parts:
            self.parts = {}
            self.part_total = 0
        self.reported = False

    def bump_part(self, part: str, count: int = 1, in_scene: bool = False):
        """Record `count` hits on `part`, ignoring non-positive amounts.

        The total tally (`parts`/`part_total`) always accumulates. The per-scene
        counters only move when `in_scene` is set, so a late "report at the end"
        call does not distort the scene subtotal.
        """
        if not part:
            return
        try:
            count = int(count)
        except (TypeError, ValueError):
            count = 1
        if count <= 0:
            return
        self.parts[part] = self.parts.get(part, 0) + count
        self.part_total += count
        if in_scene:
            self.scene_parts[part] = self.scene_parts.get(part, 0) + count
            self.scene_part_total += count


class IntimacyProgressPlugin(BasePlugin):
    def __init__(self, ctx, cfg: dict):
        super().__init__(ctx, cfg)

        # --- runtime (must exist before _apply_config touches states) ------ #
        self.states: dict[str, IntimacyState] = {}
        self._lock = asyncio.Lock()
        self._cleanup_task: asyncio.Task | None = None

        # --- config ------------------------------------------------------- #
        # All placeholders first so _apply_config never reads a missing attr.
        # Defaults here mirror schema.json: every feature off until asked for.
        self.enabled = False
        # NTR on = serve everyone; off = only the users in the whitelist.
        self.ntr = False
        self.user_whitelist: list[str] = []
        self.use_detect = False
        self.detect_keywords: list[str] = []
        self.exit_keywords: list[str] = []
        self.inject_progress = False
        self.inject_stage_hint = False
        self.force_report = False
        self.announce_tools = False
        self.auto_advance = False
        self.auto_advance_rounds = 3
        self.scene_timeout = 0.0
        self.track_desire = False
        self.allow_llm_desire = False
        self.track_parts = False
        self.parts: list[str] = list(DEFAULT_PARTS)
        self.stages: list[str] = list(DEFAULT_STAGES)
        self.stage_hints: list[str] = list(DEFAULT_STAGE_HINTS)
        self.allow_llm_reset = False
        self.proactive_end = False
        self.scene_image = False
        self.image_max_per_round = 1
        self.log_level = "info"
        self._detect_re = None
        self._exit_re = None
        self._schema: dict | None = None

        self._apply_config(dict(cfg or {}))

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _as_word_list(value) -> list[str]:
        """Normalize a config list field (may arrive as list or newline string)."""
        if not value:
            return []
        if isinstance(value, str):
            parts = re.split(r"[\n,，、]+", value)
        elif isinstance(value, (list, tuple)):
            parts = []
            for item in value:
                parts.extend(re.split(r"[\n,，、]+", str(item)))
        else:
            return []
        return [p.strip() for p in parts if p and p.strip()]

    def _log(self, msg: str, level: str = "info"):
        if level == "debug" and self.log_level != "debug":
            return
        logger.info(f"[{PLUGIN_ID}] {msg}")

    # ------------------------------------------------------------------ #
    # Config persistence
    #
    # The WebUI exposes every schema.json field as an editable control, so
    # this plugin needs to write config back to disk itself. There is no
    # public plugin-facing setter on the host, so the file layout is mirrored
    # directly: data/config/plugins/<plugin_id>.json — the same file the host
    # reads through _load_plugin_config_from_file(). Writes are atomic and
    # leave a rolling backup behind, because a corrupt config here means the
    # host cannot load the plugin at all on next boot.
    # ------------------------------------------------------------------ #

    def _config_path(self) -> str | None:
        """Resolve this plugin's config file the same way the host does."""
        try:
            from core.utils.path_utils import get_config_path
            return str(get_config_path() / "plugins" / f"{PLUGIN_ID}.json")
        except Exception as e:  # pragma: no cover - host API drift
            logger.warning(f"[{PLUGIN_ID}] cannot resolve config path: {e}")
            return None

    def _manifest_path(self) -> str | None:
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(here, MANIFEST_NAME)
        return path if os.path.isfile(path) else None

    def _read_config_file(self) -> dict:
        """Read the on-disk config, falling back to in-memory values.

        Any schema key the file is missing is filled from the schema default and
        written back. The host only seeds defaults the first time it creates the
        file, so without this a field added later would sit blank in the sidebar
        of every already-provisioned install.
        """
        path = self._config_path()
        data: dict | None = None
        if path and os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    data = loaded
                else:
                    logger.warning(f"[{PLUGIN_ID}] config file is not an object, ignoring")
            except Exception as e:
                logger.warning(f"[{PLUGIN_ID}] failed to read config file: {e}")

        if data is None:
            data = dict(self.plugin_cfg or {})

        return self._backfill_defaults(data, path)

    def _backfill_defaults(self, cfg: dict, path: str | None) -> dict:
        """Reconcile the on-disk config against the current schema.

        Missing keys are added at their default; keys the schema no longer
        defines are dropped, so a renamed or removed field cannot linger in the
        file forever. Persisted only when something actually changed.
        """
        schema = self.load_schema()
        missing = {
            k: meta.get("default")
            for k, meta in schema.items()
            if k not in cfg and isinstance(meta, dict)
        }
        # A key the schema dropped is dead weight in the sidebar and could be
        # misread as still-active config, so remove it on the next write.
        stale = [k for k in cfg if k not in schema]

        if not missing and not stale:
            return cfg

        merged = {k: v for k, v in cfg.items() if k in schema}
        merged.update(missing)
        if missing:
            self._log(f"config missing {sorted(missing)}, backfilled from schema defaults")
        if stale:
            self._log(f"config dropped unknown keys {sorted(stale)}")
        if path:
            self._write_config_file(merged)
        return merged

    def _backup_config_file(self, path: str):
        """Keep a short rolling history of config snapshots."""
        try:
            if not os.path.isfile(path):
                return
            stamp = time.strftime("%Y%m%d%H%M%S")
            shutil.copy2(path, f"{path}.bak-{stamp}")
            siblings = sorted(
                (f for f in os.listdir(os.path.dirname(path))
                 if f.startswith(f"{PLUGIN_ID}.json.bak-")),
                reverse=True,
            )
            for old in siblings[_MAX_BACKUPS:]:
                try:
                    os.remove(os.path.join(os.path.dirname(path), old))
                except OSError:
                    pass
        except Exception as e:  # pragma: no cover - best effort only
            logger.warning(f"[{PLUGIN_ID}] config backup skipped: {e}")

    def _write_config_file(self, cfg: dict) -> bool:
        """Atomically persist the config. Returns True on success."""
        path = self._config_path()
        if not path:
            return False
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self._backup_config_file(path)
            tmp = f"{path}.tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=4, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
            return True
        except Exception as e:
            logger.error(f"[{PLUGIN_ID}] failed to save config: {e}")
            try:
                os.remove(f"{path}.tmp")
            except OSError:
                pass
            return False

    # ------------------------------------------------------------------ #
    # Config: schema metadata, validation and (re)application
    # ------------------------------------------------------------------ #

    def load_schema(self) -> dict:
        """Read schema.json once — it is static metadata, not mutable state."""
        if self._schema is not None:
            return self._schema
        schema: dict = {}
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(here, "schema.json")
        try:
            if os.path.isfile(path):
                with open(path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                if isinstance(raw, dict):
                    schema = {k: v for k, v in raw.items() if isinstance(v, dict)}
        except Exception as e:
            logger.warning(f"[{PLUGIN_ID}] failed to read schema.json: {e}")
        self._schema = schema
        return schema

    def _coerce_bool(self, value, field: str) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on", "是")
        return bool(value)

    def _coerce_number(self, value, field: str, integer: bool):
        try:
            num = int(value) if integer else float(value)
        except (TypeError, ValueError):
            raise ValueError(f"「{self._field_label(field)}」需要一个数字")
        if integer:
            return max(1, num) if field == "auto_advance_rounds" else num
        return max(0.0, num) if field == "scene_timeout_minutes" else num

    def _field_label(self, field: str) -> str:
        meta = self.load_schema().get(field) or {}
        return str(meta.get("name") or field)

    def _coerce_value(self, field: str, value, meta: dict):
        """Validate/normalize one submitted value against its schema entry."""
        ftype = str(meta.get("type", "string")).lower()

        if ftype in ("boolean", "switch"):
            return self._coerce_bool(value, field)
        if ftype in ("integer", "int"):
            return self._coerce_number(value, field, True)
        if ftype in ("number", "float"):
            return self._coerce_number(value, field, False)
        if ftype == "select" or ftype == "enum":
            allowed = [
                o.get("value") for o in (meta.get("options") or [])
                if isinstance(o, dict)
            ]
            picked = str(value)
            if allowed and picked not in allowed:
                raise ValueError(
                    f"「{self._field_label(field)}」取值无效（可选：{'、'.join(map(str, allowed))}）"
                )
            return picked
        if ftype == "list":
            words = self._as_word_list(value)
            if field == "stage_names" and words and len(words) < 2:
                raise ValueError("自定义阶段至少需要 2 个，否则将回退到内置 6 阶段")
            return words
        return str(value)

    def _validate_payload(self, payload: dict) -> tuple[dict, list[str]]:
        """Coerce a partial payload. Returns (clean_updates, errors)."""
        schema = self.load_schema()
        root = payload.get("config") if isinstance(payload.get("config"), dict) else payload
        clean: dict = {}
        errors: list[str] = []
        for field, value in (root or {}).items():
            meta = schema.get(field)
            if not isinstance(meta, dict):
                # Unknown key: ignore rather than persist junk.
                continue
            if str(meta.get("type", "")).lower() == "markdown":
                continue
            try:
                clean[field] = self._coerce_value(field, value, meta)
            except ValueError as e:
                errors.append(str(e))
            except Exception as e:
                errors.append(f"「{self._field_label(field)}」无效：{e}")
        return clean, errors

    def _apply_config(self, cfg: dict):
        """Re-derive every runtime attribute from a config dict.

        Called on startup and after every WebUI save, so the running instance
        never drifts from what the file says.
        """
        self.enabled = bool(cfg.get("enabled", False))

        self.ntr = bool(cfg.get("ntr", False))
        self.user_whitelist = self._as_word_list(cfg.get("user_whitelist"))

        # Keyword detection is opt-in: with no scan there is nothing to match
        # against, so the matchers stay None and the hook short-circuits.
        self.use_detect = bool(cfg.get("use_detect", False))
        self.detect_keywords = self._as_word_list(cfg.get("detect_keywords")) \
            or list(DEFAULT_DETECT_KEYWORDS)
        self.exit_keywords = self._as_word_list(cfg.get("exit_keywords")) \
            or list(DEFAULT_EXIT_KEYWORDS)

        self.inject_progress = bool(cfg.get("inject_progress", False))

        self.auto_advance = bool(cfg.get("auto_advance", False))
        try:
            self.auto_advance_rounds = max(1, int(cfg.get("auto_advance_rounds", 3)))
        except (TypeError, ValueError):
            self.auto_advance_rounds = 3

        try:
            self.scene_timeout = max(0.0, float(cfg.get("scene_timeout_minutes", 0))) * 60.0
        except (TypeError, ValueError):
            self.scene_timeout = 0.0

        custom_stages = self._as_word_list(cfg.get("stage_names"))
        stages = custom_stages or list(DEFAULT_STAGES)
        if len(stages) < 2:
            stages = list(DEFAULT_STAGES)
        stages_changed = stages != self.stages
        self.stages = stages

        # Body-part tracker: an explicit list overrides the built-in one.
        self.track_parts = bool(cfg.get("track_parts", False))
        self.parts = self._as_word_list(cfg.get("part_names")) or list(DEFAULT_PARTS)

        # Stage hints only make sense for the built-in stage layout.
        self.stage_hints = list(DEFAULT_STAGE_HINTS) \
            if len(self.stages) == len(DEFAULT_STAGES) else []

        self.inject_stage_hint = bool(cfg.get("inject_stage_hint", False))
        self.force_report = bool(cfg.get("force_report", False))
        self.announce_tools = bool(cfg.get("announce_tools", False))
        self.track_desire = bool(cfg.get("track_desire", False))
        self.allow_llm_desire = bool(cfg.get("allow_llm_desire", False))
        self.allow_llm_reset = bool(cfg.get("allow_llm_reset", False))
        self.proactive_end = bool(cfg.get("proactive_end", False))
        self.scene_image = bool(cfg.get("scene_image", False))
        try:
            self.image_max_per_round = max(1, int(cfg.get("image_max_per_round", 1)))
        except (TypeError, ValueError):
            self.image_max_per_round = 1
        self.log_level = str(cfg.get("log_level", "info")).lower()

        # Keyword sets changed -> rebuild the precompiled matchers. Only build
        # them when detection is on; otherwise leave None so the hook can skip.
        self._detect_re = re.compile("|".join(re.escape(k) for k in self.detect_keywords)) \
            if (self.use_detect and self.detect_keywords) else None
        self._exit_re = re.compile("|".join(re.escape(k) for k in self.exit_keywords)) \
            if (self.use_detect and self.exit_keywords) else None

        # A shorter stage list can strand an active session past the last stage.
        if stages_changed:
            for st in self.states.values():
                if st.active:
                    st.stage = max(0, min(st.stage, len(self.stages) - 1))
                    st.stall_rounds = 0

        # In-memory mirror, and keep the host's copy in sync if reachable.
        self.plugin_cfg = dict(cfg)
        mgr = getattr(self.ctx, "plugin_mgr", None)
        if mgr is not None:
            try:
                mgr.plugin_configs[PLUGIN_ID] = dict(cfg)
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _scope_allows(self, event: KiraMessageBatchEvent) -> bool:
        """Whether this event's user is allowed to drive the plugin.

        NTR on → everyone. NTR off → only listed users, and an empty list means
        "nobody", so a half-configured install stays quiet instead of leaking
        the feature to everyone.
        """
        if self.ntr:
            return True
        if not self.user_whitelist:
            return False
        wanted = {w.strip().lower() for w in self.user_whitelist if w and w.strip()}
        if not wanted:
            return False
        return bool(wanted & self._identifiers(event))

    @staticmethod
    def _identifiers(event: KiraMessageBatchEvent) -> set[str]:
        """Every identity string we can see on the event, lowercased.

        Covers the sender's user id (the QQ number), the session id (group id
        in a group, user id in a DM) and the session title (nickname or group
        name), so a whitelist entry works regardless of which form the user
        happened to paste in.
        """
        out: set[str] = set()
        for m in getattr(event, "messages", None) or []:
            sender = getattr(m, "sender", None)
            uid = getattr(sender, "user_id", None)
            if uid:
                out.add(str(uid).lower())
            nick = getattr(sender, "nickname", None)
            if nick:
                out.add(str(nick).lower())
        sess = getattr(event, "session", None)
        if sess is not None:
            for attr in ("session_id", "session_title"):
                val = getattr(sess, attr, None)
                if val:
                    out.add(str(val).lower())
        return out

    @staticmethod
    def _extract_text(event: KiraMessageBatchEvent) -> str:
        """Concatenate plain text from all messages in the batch."""
        out = []
        for m in getattr(event, "messages", []) or []:
            chain = getattr(m, "chain", None)
            if chain is None:
                continue
            for ele in chain:
                if isinstance(ele, Text):
                    out.append(ele.text or "")
                else:
                    text = getattr(ele, "text", None)
                    if isinstance(text, str):
                        out.append(text)
        return "\n".join(out)

    def _stage_name(self, idx: int) -> str:
        idx = max(0, min(idx, len(self.stages) - 1))
        return self.stages[idx]

    def _state(self, sid: str) -> IntimacyState:
        """Fetch (or lazily create) the per-session state."""
        st = self.states.get(sid)
        if st is None:
            st = IntimacyState(sid)
            self.states[sid] = st
        return st

    def _is_last_stage(self, idx: int) -> bool:
        return idx >= len(self.stages) - 1

    def _timeout_expired(self, st: IntimacyState) -> bool:
        if self.scene_timeout <= 0:
            return False
        return (time.time() - st.last_active) > self.scene_timeout

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    async def initialize(self):
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
        if self.scene_timeout > 0:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        self._log(
            f"initialized | stages={self.stages} "
            f"| ntr={self.ntr} users={len(self.user_whitelist)} "
            f"| use_detect={self.use_detect} | auto_advance={self.auto_advance}"
            f"({self.auto_advance_rounds})"
        )
        # Say up front whether the image feature can actually work — a missing
        # default image model is otherwise only discovered at generation time.
        if self.scene_image:
            try:
                self.ctx.provider_mgr.get_default_image()
                self._log(
                    f"scene_image on | default image model OK "
                    f"| max {self.image_max_per_round} per round"
                )
            except Exception as e:
                self._log(
                    f"scene_image is on but no usable default image model: {e} "
                    f"— configure one in KiraAI 设置, or AI won't be able to draw.",
                    level="warning",
                )

    async def terminate(self):
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except (asyncio.CancelledError, Exception):
                pass
        self._cleanup_task = None
        self.states.clear()
        self._log("terminated")

    async def _cleanup_loop(self):
        """Periodically drop sessions whose scene has gone cold."""
        try:
            while True:
                await asyncio.sleep(60)
                async with self._lock:
                    stale = [
                        sid for sid, st in self.states.items()
                        if st.active and self._timeout_expired(st)
                    ]
                    for sid in stale:
                        self.states[sid].reset()
                        self._log(f"session {sid} scene timed out, progress reset")
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(f"[{PLUGIN_ID}] cleanup loop error")

    # ------------------------------------------------------------------ #
    # Stage transition primitives
    # ------------------------------------------------------------------ #

    def _begin(self, st: IntimacyState):
        st.active = True
        st.stage = 0
        st.rounds = 0
        st.total_rounds = 0
        st.stall_rounds = 0
        st.started_at = time.time()
        st.last_active = time.time()
        st.desire = DESIRE_ON_BEGIN if self.track_desire else 0
        # A new scene owes a fresh report, and counts its own parts.
        st.reported = False
        st.scene_parts = {}
        st.scene_part_total = 0
        st.images_this_round = 0

    def _advance(self, st: IntimacyState, steps: int = 1) -> bool:
        """Move forward N stages. Returns False if already at the last stage."""
        if self._is_last_stage(st.stage):
            return False
        st.stage = min(st.stage + steps, len(self.stages) - 1)
        st.rounds = 0
        st.stall_rounds = 0
        # The stage moved, so the previous acknowledgement no longer covers it.
        st.reported = False
        if self.track_desire:
            st.desire = min(100, st.desire + DESIRE_ON_ADVANCE * steps)
        return True

    def _retreat(self, st: IntimacyState, steps: int = 1) -> bool:
        if st.stage <= 0:
            return False
        st.stage = max(0, st.stage - steps)
        st.rounds = 0
        st.stall_rounds = 0
        st.reported = False
        if self.track_desire:
            st.desire = max(0, st.desire - DESIRE_ON_RETREAT * steps)
        return True

    # ------------------------------------------------------------------ #
    # Prompt building
    # ------------------------------------------------------------------ #

    def _build_status_block(self, st: IntimacyState) -> str:
        idx = max(0, min(st.stage, len(self.stages) - 1))
        name = self._stage_name(idx)
        total = len(self.stages)

        lines = [
            "## 亲密场景进度（Intimacy Progress）",
            "",
            "你与对方正处于亲密场景中。以下进度用于帮助你把握叙事节奏——",
            "**你清楚现在进行到哪一步，请让剧情自然地停在这一步的合理范围内，不要原地打转，也不要提前跳到后面的阶段。**",
            "",
            f"- 当前阶段：**第 {idx + 1} / {total} 阶段 — 「{name}」**",
            f"- 该阶段已持续：{st.rounds} 轮",
            f"- 场景累计：{st.total_rounds} 轮",
        ]

        if self.track_desire:
            feel = self._desire_label(st.desire)
            lines.append(f"- 当前兴奋度：{int(st.desire)}/100（{feel}）")

        if self.track_parts and st.parts:
            ranked = sorted(st.parts.items(), key=lambda kv: (-kv[1], kv[0]))
            shown = "、".join(f"{p}×{n}" for p, n in ranked[:8])
            lines.append(f"- 分部位次数（本场累计 {st.part_total} 次）：{shown}")

        # Progress bar for a quick visual read.
        bar_len = 12
        filled = int(round((idx + 1) / total * bar_len))
        bar = "█" * filled + "░" * (bar_len - filled)
        lines.append(f"- 进度：`{bar}` {name}")

        # Pacing reminders driven by state.
        hints = []
        if st.rounds >= self.auto_advance_rounds + 1:
            hints.append("你已经在这个阶段停留了较久，可以自然地向前推进一点了。")
        if self._is_last_stage(idx):
            if self.proactive_end:
                # Escalate with how long the scene has been winding down, so the
                # model actually closes instead of trailing off indefinitely.
                if st.rounds >= self.auto_advance_rounds:
                    hints.append(
                        f"**你已经在最后阶段停留了 {st.rounds} 轮，该收尾了。**"
                        "请在这一轮写出一个自然、舒缓的结尾，把场景收束掉，"
                        "不要再用新的情节或新的话题把它拉长。"
                    )
                else:
                    hints.append(
                        "这已经是最后阶段，请往收尾的方向写，"
                        "留出余韵即可，不要再推进新的情节。"
                    )
            else:
                hints.append("这已经是最后阶段。请在合适的时机自然地收束场景，不要无限拖长。")
        if idx == 0 and st.total_rounds == 0:
            hints.append("场景刚刚开始，先营造气氛，不要急着发生实质接触。")

        if self.inject_stage_hint and self.stage_hints:
            hint = self.stage_hints[min(idx, len(self.stage_hints) - 1)]
            hints.append(f"本阶段节奏参考：{hint}")

        if hints:
            lines.append("")
            lines.append("### 节奏提示")
            for h in hints:
                lines.append(f"- {h}")

        # Forced reporting: make the acknowledgement an explicit output every
        # round, so the progress state cannot be silently ignored.
        if self.force_report:
            lines.append("")
            lines.append("### 本轮必须回报（强制）")
            if st.reported:
                lines.append(
                    "- 你这一轮已经回报过了，**不要重复调用** `int_report`，直接继续写剧情即可。"
                )
            else:
                lines.append(
                    "- **本轮你必须先调用 `int_report` 工具**回报一次当前进度，然后再开始写正文。"
                    "这是每轮都要做的，不是可选项。"
                )
                lines.append(
                    "- `int_report` 是一个**函数工具**，请通过**工具调用**提交。"
                    "**绝对不要**在消息正文里写 `<report .../>`、`&lt;report .../&gt;` "
                    "或任何标签形式的回报——那会被当成发给对方的消息，对方只会看到一串乱码。"
                )
                lines.append(
                    "- `int_report` 的 `progress` 就是你判断的当前阶段名称"
                    f"（此刻应为「{name}」）；`note` 用一句话说明你这一轮做了什么、"
                    "以及下一步打算怎么走。"
                )
                lines.append(
                    "- 如果你认为剧情已经可以更进一步，就在这次汇报里说明理由，"
                    "并同时调用 `int_advance`；系统会把新阶段写进下一轮的进度里。"
                )
                lines.append(
                    "- 回报之后，正文请严格停在你汇报的那个阶段范围内，不要提前跳到后面的阶段。"
                )

        if self.scene_image:
            left = max(0, self.image_max_per_round - st.images_this_round)
            lines.append("")
            lines.append("### 场景配图")
            if left > 0:
                lines.append(
                    f"- **本回合你要生成配图**：用 `<img>画面描述</img>` 输出"
                    f"（走系统配置的生图模型，会自动发给对方）。"
                )
                lines.append(
                    f"- 本回合上限 {self.image_max_per_round} 张，你还剩 **{left}** 张额度。"
                    "至少生成一张给这一段画面定格。"
                )
            else:
                lines.append(
                    "- **本回合的配图额度已经用完，不要再生成图片了**，继续用文字推进即可。"
                )
            lines.append(
                "- 画面描述要写具体：镜头、构图、光线、表情、姿态、衣着状态等；"
                "贴合当前阶段和刚刚发生的情节，不要和上一张重复。"
            )

        lines.append("")
        lines.append("### 你可以主动控制进度")
        lines.append("- 想推进时，调用 `int_advance` 工具（可传 `reason` 说明理由）。")
        lines.append("- 想退回上一阶段（例如对方抗拒、气氛冷却），调用 `int_retreat`。")
        if self.allow_llm_desire and self.track_desire:
            lines.append(
                "- 对方情绪有明显波动时，可调 `int_desire(delta, reason?)` 直接改兴奋度"
                "（正数升温、负数降温）。小幅度即可，不要频繁调用。"
            )
        if self.track_parts:
            parts = "、".join(self.parts[:10])
            lines.append(
                f"- **场景结束或收尾时**，调用 `int_part(part, count)` 汇报本场各部位的累计次数"
                f"（可用部位：{parts}）。可以在收尾那一轮里分几次调用，把发生过的部位都报一遍；"
                "不需要在每一轮中途记录。"
            )
        if self.allow_llm_reset:
            if self.proactive_end:
                lines.append(
                    "- **判断该收尾时，主动调用 `int_reset` 结束场景**，不必等对方先开口。"
                    "一直拖着不收尾比收得早一点更糟。"
                )
                if self.track_parts:
                    lines.append(
                        "- 收尾时记得先用 `int_part` 汇报本场各部位次数，再调 `int_reset`。"
                    )
            else:
                lines.append("- 场景自然结束或对方明确结束后，调用 `int_reset`（或用户说了结束语时系统会自动重置）。")
        lines.append("- 不需要每轮都调用；只有在节奏确实需要变化时才调用。")

        return "\n".join(lines)

    @staticmethod
    def _desire_label(value: float) -> str:
        if value < 15:
            return "平静"
        if value < 40:
            return "微热"
        if value < 65:
            return "升温"
        if value < 85:
            return "炽热"
        return "濒临失控"

    # ------------------------------------------------------------------ #
    # Detection
    # ------------------------------------------------------------------ #

    def _detect_entry(self, text: str) -> str | None:
        if not text or not self._detect_re:
            return None
        m = self._detect_re.search(text)
        return m.group(0) if m else None

    def _detect_exit(self, text: str) -> str | None:
        if not text or not self._exit_re:
            return None
        m = self._exit_re.search(text)
        return m.group(0) if m else None

    # ------------------------------------------------------------------ #
    # Hooks
    # ------------------------------------------------------------------ #

    @on.im_batch_message(priority=Priority.MEDIUM)
    async def on_batch_message(self, event: KiraMessageBatchEvent, *_):
        if not self.enabled or not self._scope_allows(event):
            return

        sid = event.sid
        text = self._extract_text(event)

        async with self._lock:
            st = self._state(sid)

            # 1) Explicit exit keywords always win.
            hit_exit = self._detect_exit(text)
            if hit_exit is not None and st.active:
                st.reset()
                self._log(f"session {sid} exit detected by '{hit_exit}', progress reset")
                return

            # 2) Auto entry.
            if not st.active and self.use_detect:
                hit = self._detect_entry(text)
                if hit is not None:
                    self._begin(st)
                    self._log(
                        f"session {sid} scene started by '{hit}' "
                        f"-> stage 「{self._stage_name(st.stage)}」"
                    )
                return

            if not st.active:
                return

            # 3) Advance the round counters for an ongoing scene.
            if self._timeout_expired(st):
                st.reset()
                self._log(f"session {sid} scene timed out on access, progress reset")
                return

            st.total_rounds += 1
            st.rounds += 1
            st.last_active = time.time()
            # A new user turn starts a new round, which owes a new report and
            # gets a fresh image budget.
            st.reported = False
            st.images_this_round = 0

            if self.track_desire and not self._is_last_stage(st.stage):
                st.desire = min(100, st.desire + DESIRE_PER_ROUND)

    @on.llm_request(priority=Priority.MEDIUM)
    async def hook_inject_progress(self, event: KiraMessageBatchEvent, req: LLMRequest, *_):
        # NOTE: the method name deliberately does NOT match the `inject_progress`
        # config flag. The host binds a plugin hook via
        # `getattr(instance, func.__name__)`, so naming the method after the flag
        # would make that getattr return the bool and bind a bool as the handler —
        # silently killing the hook.
        if not self.enabled or not getattr(self, "inject_progress", False):
            return
        if not self._scope_allows(event):
            return

        sid = event.sid
        st = self.states.get(sid)
        if st is None or not st.active:
            return

        if self._timeout_expired(st):
            return

        # Auto-advance: if the model has been idling, nudge the stage forward so
        # the scene cannot stall forever. Explicit tool calls reset stall_rounds.
        if self.auto_advance and st.stall_rounds >= self.auto_advance_rounds:
            if self._advance(st, 1):
                self._log(
                    f"session {sid} auto-advanced to "
                    f"「{self._stage_name(st.stage)}」 after {self.auto_advance_rounds} idle rounds"
                )
            else:
                # Already at the end: no more auto-advancing.
                st.stall_rounds = 0

        # Count this round as idle until a tool call proves otherwise.
        st.stall_rounds += 1

        block = self._build_status_block(st)
        req.system_prompt.append(Prompt(
            content=block,
            name="intimacy_progress",
            source=PLUGIN_ID,
        ))
        self._log(
            f"injected progress for {sid} | stage={st.stage + 1}/{len(self.stages)} "
            f"「{self._stage_name(st.stage)}」 rounds={st.rounds}",
            level="debug",
        )

    # ------------------------------------------------------------------ #
    # Tools — the LLM drives the pacing
    # ------------------------------------------------------------------ #

    @register.tool(
        name="int_advance",
        description=(
            "推进亲密场景到下一个阶段。当前没有进行中的场景时，调用它会直接开启一段新场景"
            "（从第一阶段开始）。只有当剧情自然发展到可以更进一步时才调用；"
            "不要连续调用，也不要在还没铺垫好时急着推进。"
        ),
        params={
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "推进的理由，例如「对方已经放松下来，开始回应」",
                },
                "steps": {
                    "type": "integer",
                    "description": "一次推进几个阶段，默认 1。通常保持 1。",
                },
            },
            "required": [],
        },
    )
    async def int_advance(self, event: KiraMessageBatchEvent, *_, reason: str = "", steps: int = 1) -> str:
        if not self.enabled:
            return "插件当前未启用。"
        sid = event.sid
        async with self._lock:
            st = self._state(sid)
            opened = False
            if not st.active:
                # Allow the model to open a scene explicitly. Opening already
                # lands on stage 1, so do NOT also advance — otherwise the first
                # stage would be skipped and the scene would start at 升温.
                self._begin(st)
                opened = True
                self._log(f"session {sid} scene explicitly started via int_advance")

            try:
                steps = max(1, int(steps))
            except (TypeError, ValueError):
                steps = 1

            moved = self._advance(st, steps) if not opened else True
            st.last_active = time.time()
            name = self._stage_name(st.stage)
            idx = st.stage + 1

        if not moved:
            return f"已经处于最后阶段「{name}」，无法继续推进。请开始收束场景。"
        self._log(f"session {sid} advanced to 「{name}」 ({idx}/{len(self.stages)}) reason={reason!r}")
        tail = ""
        if self._is_last_stage(st.stage):
            tail = ("（这是最后阶段，请往收尾的方向写，"
                    "写出余韵后主动调用 `int_reset` 结束。）"
                    if self.proactive_end else "（这是最后阶段，注意准备收尾。）")
        if opened:
            return (
                f"已开启亲密场景，当前是第 {idx}/{len(self.stages)} 阶段「{name}」。"
                "接下来每轮你都会收到进度提示。" + tail
            )
        return f"已推进到第 {idx}/{len(self.stages)} 阶段「{name}」。{tail}"

    @register.tool(
        name="int_retreat",
        description=(
            "退回亲密场景的上一个阶段。当对方表现出抗拒、犹豫，或气氛冷下来时使用。"
        ),
        params={
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "退回的理由",
                },
                "steps": {
                    "type": "integer",
                    "description": "一次退回几个阶段，默认 1。",
                },
            },
            "required": [],
        },
    )
    async def int_retreat(self, event: KiraMessageBatchEvent, *_, reason: str = "", steps: int = 1) -> str:
        if not self.enabled:
            return "插件当前未启用。"
        sid = event.sid
        async with self._lock:
            st = self._state(sid)
            if not st.active:
                return "当前没有进行中的亲密场景。"
            try:
                steps = max(1, int(steps))
            except (TypeError, ValueError):
                steps = 1
            moved = self._retreat(st, steps)
            st.last_active = time.time()
            name = self._stage_name(st.stage)
            idx = st.stage + 1

        if not moved:
            return "已经在最初阶段，无法再退回。"
        self._log(f"session {sid} retreated to 「{name}」 ({idx}/{len(self.stages)}) reason={reason!r}")
        return f"已退回到第 {idx}/{len(self.stages)} 阶段「{name}」。"

    @register.tool(
        name="int_status",
        description=(
            "查看当前亲密场景的进度状态。一般不需要调用，系统已自动注入；"
            "仅在你不确定有没有场景在跑时使用。"
        ),
        params={"type": "object", "properties": {}, "required": []},
    )
    async def int_status(self, event: KiraMessageBatchEvent, *_) -> str:
        if not self.enabled:
            return "插件当前未启用。"
        sid = event.sid
        st = self._state(sid)
        if not st.active:
            return (
                "当前没有进行中的亲密场景。"
                "如果你判断现在适合开始，可以调用 `int_advance` 主动开启（会从第一阶段开始）。"
            )
        idx = st.stage + 1
        total = len(self.stages)
        desire = f"，兴奋度 {int(st.desire)}/100" if self.track_desire else ""
        parts = ""
        if self.track_parts and st.parts:
            ranked = sorted(st.parts.items(), key=lambda kv: (-kv[1], kv[0]))
            parts = "；分部位：" + "、".join(f"{p}×{n}" for p, n in ranked[:8])
        return (
            f"当前处于第 {idx}/{total} 阶段「{self._stage_name(st.stage)}」，"
            f"该阶段已 {st.rounds} 轮，场景累计 {st.total_rounds} 轮{desire}{parts}。"
        )

    @register.tool(
        name="int_part",
        description=(
            "记录某个部位被碰触/刺激的次数，用于侧边栏的分部位统计。"
            "**通常在一场场景结束或收尾时，一次性汇报本场累计的次数**"
            "（同一次调用可以只报一个部位，也可以分多次报不同部位）。"
            "不要为了凑数而调用，只记录确实发生过的。"
        ),
        params={
            "type": "object",
            "properties": {
                "part": {
                    "type": "string",
                    "description": "部位名称，尽量用配置里的部位词，例如「唇」「胸」「私处」。",
                },
                "count": {
                    "type": "integer",
                    "description": "本场累计次数，默认 1。收尾汇报时填这一场发生的总次数。",
                },
            },
            "required": ["part"],
        },
    )
    async def int_part(self, event: KiraMessageBatchEvent, *_,
                       part: str = "", count: int = 1) -> str:
        if not self.enabled:
            return "插件当前未启用。"
        if not self.track_parts:
            return "当前配置未开启分部位统计。"
        part = (part or "").strip()
        if not part:
            return "需要提供部位名称。"

        sid = event.sid
        async with self._lock:
            st = self._state(sid)
            # A wrap-up report lands right after the scene ends, so do not
            # require an active scene — but only count it toward the scene
            # subtotal when the scene really is still running.
            st.bump_part(part, count, in_scene=st.active)
            st.last_active = time.time()
            total = st.parts.get(part, 0)
            scene_total = st.part_total
            scene_sub = st.scene_part_total

        self._log(
            f"session {sid} part 「{part}」 +{count} -> {total} "
            f"(累计 {scene_total}，本场 {scene_sub})"
        )
        return f"已记录「{part}」+{count}，该部位累计 {total} 次（总计 {scene_total} 次）。"

    @register.tool(
        name="int_desire",
        description=(
            "直接调整当前兴奋度（desire）。传正数增加、负数降低，"
            "用于表达剧情里对方情绪的明显变化，例如对方突然害羞降温、"
            "或被挑逗得骤然升温。仅在开启「允许 AI 控制兴奋度」后可用。"
        ),
        params={
            "type": "object",
            "properties": {
                "delta": {
                    "type": "integer",
                    "description": "兴奋度的增减量，正数增加、负数降低，范围约 -40 ~ 40。",
                },
                "reason": {
                    "type": "string",
                    "description": "调整的理由，例如「对方突然害羞」「被咬到耳垂」",
                },
            },
            "required": ["delta"],
        },
    )
    async def int_desire(self, event: KiraMessageBatchEvent, *_,
                         delta: int = 0, reason: str = "") -> str:
        if not self.enabled:
            return "插件当前未启用。"
        if not self.track_desire:
            return "当前配置未开启兴奋度追踪。"
        if not self.allow_llm_desire:
            return "当前配置未允许 AI 控制兴奋度。"
        try:
            delta = int(delta)
        except (TypeError, ValueError):
            return "需要提供有效的增减量。"
        # Guard against absurd values the model might invent.
        delta = max(-40, min(40, delta))

        sid = event.sid
        async with self._lock:
            st = self._state(sid)
            if not st.active:
                return "当前没有进行中的亲密场景，未记录。"
            before = st.desire
            st.desire = max(0.0, min(100.0, st.desire + delta))
            st.last_active = time.time()
            after = st.desire

        feel = self._desire_label(after)
        self._log(f"session {sid} desire {delta:+d} -> {int(after)}/100 reason={reason!r}")
        return f"兴奋度已调整 {delta:+d}，当前 {int(after)}/100（{feel}）。"

    @register.tool(
        name="int_report",
        description=(
            "回报本轮亲密场景进度。开启「强制每轮回报」后，每轮都必须先调用一次，"
            "再说/写别的；未开启时只在进度有变化时调用即可。"
        ),
        params={
            "type": "object",
            "properties": {
                "progress": {
                    "type": "string",
                    "description": "你判断当前所处的阶段名称，例如「前戏」。",
                },
                "note": {
                    "type": "string",
                    "description": "一句话说明这一轮做了什么、下一步打算怎么走。",
                },
            },
            "required": [],
        },
    )
    async def int_report(self, event: KiraMessageBatchEvent, *_,
                         progress: str = "", note: str = "") -> str:
        if not self.enabled:
            return "插件当前未启用。"
        if not self.force_report:
            return "当前配置未开启强制每轮回报。"
        sid = event.sid
        async with self._lock:
            st = self._state(sid)
            if not st.active:
                return "当前没有进行中的亲密场景，未记录。"
            st.reported = True
            st.last_active = time.time()
            idx = max(0, min(st.stage, len(self.stages) - 1))
            name = self._stage_name(idx)

        # A mismatch is worth surfacing: it is the model's own read of the scene
        # against the plugin's counter, and it tells the model whether to move.
        said = (progress or "").strip()
        verdict = ""
        if said and said != name:
            if said in self.stages:
                verdict = (
                    f"（你报的是「{said}」，与系统记录的「{name}」不一致。"
                    "如果确实该推进了，请调用 `int_advance`；如果只是感觉如此，以系统进度为准。）"
                )
            else:
                verdict = f"（「{said}」不在阶段列表里，以系统记录的「{name}」为准。）"

        self._log(
            f"session {sid} report | said={said or '-'} stage={idx + 1}/{len(self.stages)}"
            f"「{name}」 note={note!r}",
            level="debug",
        )
        return f"已记录本轮回报：当前是第 {idx + 1}/{len(self.stages)} 阶段「{name}」。{verdict}"

    @register.tool(
        name="int_reset",
        description=(
            "结束/重置当前亲密场景进度。当对方明确表示结束、场景已经自然收尾、"
            "你自己判断这一段该收尾了，或需要开启一段全新的场景时调用。"
        ),
        params={
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "结束场景的理由",
                },
            },
            "required": [],
        },
    )
    async def int_reset(self, event: KiraMessageBatchEvent, *_, reason: str = "") -> str:
        if not self.enabled:
            return "插件当前未启用。"
        if not self.allow_llm_reset:
            return "当前配置不允许重置场景。"
        sid = event.sid
        async with self._lock:
            st = self._state(sid)
            st.reset()
        self._log(f"session {sid} scene reset via int_reset reason={reason!r}")
        return "已重置亲密场景进度。下一次进入时会从第一阶段重新开始。"

    @on.llm_request(priority=Priority.LOW)
    async def append_tool_doc(self, event: KiraMessageBatchEvent, req: LLMRequest, *_):
        """Append a short usage note to the tools prompt section.

        Two modes:

        * Scene active — the note lists the tools that are live right now, so the
          model knows how to steer a scene already in progress.
        * No scene yet (`announce_tools` on) — the note introduces the plugin and
          tells the model it may open a scene itself. Without this the feature is
          invisible: the model cannot call a tool it has never been told about,
          and a scene only starts from a tool call or a keyword hit.
        """
        if not self.enabled or not getattr(self, "inject_progress", False):
            return
        if not self._scope_allows(event):
            return

        st = self.states.get(event.sid)
        active = st is not None and st.active

        # Not only the model must know these tools exist. When nothing is running,
        # stay quiet unless the user asked us to announce them.
        if not active and not self.announce_tools:
            return

        if active:
            note = (
                "\n### 亲密场景进度工具\n"
                "- `int_advance(reason?, steps?)`：推进到下一阶段。剧情真的更进一步时才调用，不要连点。\n"
                "- `int_retreat(reason?, steps?)`：退回上一阶段。对方抗拒或气氛冷却时使用。\n"
                "- `int_status()`：查看进度（通常无需调用，进度已自动注入）。\n"
            )
            if self.force_report:
                note += (
                    "- `int_report(progress?, note?)`：回报本轮进度。"
                    "**开启「强制每轮回报」后，每轮都必须先调用它，再写正文。**\n"
                )
            if self.track_parts:
                note += (
                    "- `int_part(part, count)`：记录某部位次数。"
                    "**在场景结束/收尾时汇报本场累计次数**，中途不用记。\n"
                )
            if self.allow_llm_desire and self.track_desire:
                note += (
                    "- `int_desire(delta, reason?)`：直接调整兴奋度，"
                    "正数升温、负数降温。只在情绪明显变化时用，小幅度即可。\n"
                )
            if self.allow_llm_reset:
                if self.proactive_end:
                    note += (
                        "- `int_reset(reason?)`：结束并重置场景。"
                        "**你判断该收尾时就主动调用，不必等对方先开口**；"
                        "收尾前先用 `int_part` 把本场次数报完。\n"
                    )
                else:
                    note += "- `int_reset(reason?)`：结束并重置场景。\n"
        else:
            # Nothing running: introduce the feature and how to switch it on.
            stage_list = " → ".join(self.stages)
            note = (
                "\n### 亲密场景进度工具（当前未开启场景）\n"
                "你具备一套「亲密场景进度」能力：它把当前阶段、轮次写进你的上下文，"
                "帮助你把握叙事节奏——该慢的时候慢下来，该推进的时候推进，"
                "而不是原地打转或一口气跳到结尾。\n\n"
                f"阶段划分：{stage_list}\n\n"
                "当对话自然地朝亲密方向发展、或对方主动挑明时，你可以调用 "
                "`int_advance(reason?)` **主动开启场景**（会从第一阶段开始）；"
                "之后每一轮都会自动收到进度提示。\n"
                "如果对方明确停下或场景结束，调用 `int_reset(reason?)` 收尾。\n"
                "不确定当前状态时可以调 `int_status()` 查看。\n"
                "**当前没有进行中的场景，所以不要回报进度、也不要输出任何进度标签；**"
                "只有真的进入场景之后才需要那套动作。\n"
                "**不要在气氛还不够时急着调用**；没有合适时机就什么都不用做。\n"
            )
            if self.track_parts:
                note += (
                    "- `int_part(part, count)`：记录某部位次数，"
                    "在场景收尾时汇报本场累计。\n"
                )

        for p in req.system_prompt:
            if p.name == "tools":
                p.content += note
                break

    @on.after_xml_parse(priority=Priority.LOW)
    async def hook_strip_leaked_markup(self, event: KiraMessageBatchEvent, actions: list, *_):
        """Remove tool-call markup the model leaked into message text.

        An LLM asked to "call int_report" sometimes emits a literal
        ``<report .../>`` (escaped, because it sits inside ``<text>``) rather
        than making a real function call. That string would otherwise be sent
        to the user verbatim. We can't un-send it from the model's side, but we
        can make sure it never reaches the chat.
        """
        if not self.enabled or not self._scope_allows(event):
            return

        stripped = 0
        for action in actions:
            chain = getattr(action, "message_list", None)
            if not isinstance(chain, list):
                continue
            for ele in chain:
                if not isinstance(ele, Text):
                    continue
                body = ele.text
                if not body:
                    continue
                cleaned = _LEAKED_TOOL_MARKUP_RE.sub("", body).strip()
                if cleaned != body:
                    ele.text = cleaned
                    stripped += 1

        if stripped:
            self._log(
                f"session {event.sid} stripped leaked tool markup from "
                f"{stripped} message element(s)"
            )

    @on.after_xml_parse(priority=Priority.MEDIUM)
    async def hook_count_scene_images(self, event: KiraMessageBatchEvent, actions: list, *_):
        """Tally (and cap) the images the model emitted this turn.

        Image generation happens inside the `<img>` tag handler, i.e. *before*
        this stage — so by the time we see the pictures the compute is already
        spent. The real defence is the per-round budget spelled out in the
        prompt; this is the enforcement backstop: anything past the budget is
        dropped so the chat cannot be flooded.
        """
        if not self.enabled or not self.scene_image:
            return
        if not self._scope_allows(event):
            return
        st = self.states.get(event.sid)
        if st is None or not st.active:
            return

        budget = self.image_max_per_round - st.images_this_round
        kept = 0
        dropped = 0

        async with self._lock:
            for action in actions:
                # Only message chains carry renderable elements; root-tag
                # actions are left untouched.
                chain = getattr(action, "message_list", None)
                if not isinstance(chain, list):
                    continue
                remaining = []
                for ele in chain:
                    if isinstance(ele, Image):
                        if budget - kept > 0:
                            kept += 1
                            remaining.append(ele)
                        else:
                            dropped += 1
                        continue
                    remaining.append(ele)
                if dropped:
                    action.message_list = remaining

            st.images_this_round += kept

        if kept or dropped:
            self._log(
                f"session {event.sid} images | kept={kept} dropped={dropped} "
                f"(round budget {self.image_max_per_round}, used {st.images_this_round})"
            )

    # ------------------------------------------------------------------ #
    # WebUI: sidebar page + JSON API
    # ------------------------------------------------------------------ #

    def snapshot(self) -> dict:
        """Serialize plugin config + all session states for the WebUI."""
        now = time.time()
        sessions = []
        for sid, st in self.states.items():
            if not st.active:
                continue
            idx = max(0, min(st.stage, len(self.stages) - 1))

            # Per-part tally, richest first, so the sidebar can render it as-is.
            parts = [
                {"part": p, "count": n}
                for p, n in sorted(st.parts.items(), key=lambda kv: (-kv[1], kv[0]))
            ]

            sessions.append({
                "sid": sid,
                "adapter": sid.split(":", 1)[0] if ":" in sid else sid,
                "stage": idx + 1,
                "stage_name": self._stage_name(idx),
                "rounds": st.rounds,
                "total_rounds": st.total_rounds,
                "desire": int(st.desire),
                "desire_label": self._desire_label(st.desire) if self.track_desire else "",
                "idle_seconds": int(now - st.last_active) if st.last_active else 0,
                "elapsed_seconds": int(now - st.started_at) if st.started_at else 0,
                "parts": parts,
                "part_total": st.part_total,
                "scene_part_total": st.scene_part_total,
                "reported": bool(st.reported),
            })
        sessions.sort(key=lambda s: s["idle_seconds"])

        # The sidebar renders every schema field as an editable control, so the
        # payload carries the raw config, the schema metadata, and the section
        # grouping the page uses to lay the form out.
        schema = self.load_schema()
        config = self._read_config_file()

        return {
            "enabled": self.enabled,
            "ntr": self.ntr,
            "user_whitelist": list(self.user_whitelist),
            "use_detect": self.use_detect,
            "inject_progress": getattr(self, "inject_progress", False),
            "inject_stage_hint": self.inject_stage_hint,
            "force_report": self.force_report,
            "announce_tools": self.announce_tools,
            "allow_llm_reset": self.allow_llm_reset,
            "proactive_end": self.proactive_end,
            "scene_image": self.scene_image,
            "image_max_per_round": self.image_max_per_round,
            "auto_advance": self.auto_advance,
            "auto_advance_rounds": self.auto_advance_rounds,
            "scene_timeout_minutes": round(self.scene_timeout / 60.0, 1),
            "track_desire": self.track_desire,
            "allow_llm_desire": self.allow_llm_desire,
            "log_level": self.log_level,
            "stages": list(self.stages),
            "default_stages": list(DEFAULT_STAGES),
            "track_parts": self.track_parts,
            "parts": list(self.parts),
            "active_count": len(sessions),
            "config": config,
            "schema": schema,
            "sections": [
                {"key": k, "label": label, "fields": fields, "collapsed": collapsed}
                for k, label, fields, collapsed in CONFIG_SECTIONS
            ],
            "config_path": self._config_path(),
            "sessions": sessions,
        }

    @register.page(
        "/index",
        menu=PageMenu(
            label={"zh": "草傻子专用", "en": "Caoshazi"},
            icon="DataLine",
            order=60,
        ),
    )
    def manager_page(self):
        return PluginPage.from_folder("./web")

    @register.api(method="GET", path="/status")
    async def api_status(self):
        """Current plugin config + live session states."""
        return self.snapshot()

    @register.api(method="POST", path="/advance")
    async def api_advance(self, payload: dict):
        """Manually advance a session's stage (WebUI control)."""
        sid = (payload or {}).get("sid", "")
        try:
            steps = max(1, int((payload or {}).get("steps", 1)))
        except (TypeError, ValueError):
            steps = 1
        async with self._lock:
            st = self.states.get(sid)
            if st is None or not st.active:
                return {"ok": False, "error": "该会话没有进行中的场景"}
            moved = self._advance(st, steps)
            st.last_active = time.time()
            name = self._stage_name(st.stage)
            idx = st.stage + 1
        if not moved:
            return {"ok": False, "error": f"已是最后阶段「{name}」"}
        self._log(f"WebUI advanced {sid} to 「{name}」 ({idx}/{len(self.stages)})")
        return {"ok": True, "stage": idx, "stage_name": name}

    @register.api(method="POST", path="/retreat")
    async def api_retreat(self, payload: dict):
        """Manually retreat a session's stage (WebUI control)."""
        sid = (payload or {}).get("sid", "")
        try:
            steps = max(1, int((payload or {}).get("steps", 1)))
        except (TypeError, ValueError):
            steps = 1
        async with self._lock:
            st = self.states.get(sid)
            if st is None or not st.active:
                return {"ok": False, "error": "该会话没有进行中的场景"}
            moved = self._retreat(st, steps)
            st.last_active = time.time()
            name = self._stage_name(st.stage)
            idx = st.stage + 1
        if not moved:
            return {"ok": False, "error": "已经在最初阶段"}
        self._log(f"WebUI retreated {sid} to 「{name}」 ({idx}/{len(self.stages)})")
        return {"ok": True, "stage": idx, "stage_name": name}

    @register.api(method="POST", path="/reset_parts")
    async def api_reset_parts(self, payload: dict):
        """Clear the body-part tally for one session without ending the scene."""
        sid = (payload or {}).get("sid", "")
        async with self._lock:
            st = self.states.get(sid)
            if st is None:
                return {"ok": False, "error": "没有该会话的记录"}
            cleared = st.part_total
            st.parts = {}
            st.part_total = 0
            st.scene_parts = {}
            st.scene_part_total = 0
        self._log(f"WebUI cleared part tally for {sid} ({cleared} hits)")
        return {"ok": True, "cleared": cleared}

    @register.api(method="POST", path="/reset")
    async def api_reset(self, payload: dict):
        """End the running scene(s).

        Deliberately narrow: only the scene state is cleared (stage, rounds,
        desire). The body-part tally is a profile of the person and survives,
        so re-entering a scene does not lose their history.
        """
        sid = (payload or {}).get("sid", "")
        async with self._lock:
            if not sid or sid == "all":
                count = sum(1 for st in self.states.values() if st.active)
                for st in self.states.values():
                    st.reset()
                self._log(f"WebUI reset all sessions ({count} active)")
                return {"ok": True, "reset": count}
            st = self.states.get(sid)
            if st is None or not st.active:
                return {"ok": False, "error": "该会话没有进行中的场景"}
            st.reset()
        self._log(f"WebUI reset {sid}")
        return {"ok": True, "reset": 1}

    @register.api(method="POST", path="/config")
    async def api_config(self, payload: dict):
        """Validate, persist and hot-apply a partial config update.

        Returns the saved config so the sidebar can re-render from the
        authoritative values rather than what the user typed.
        """
        payload = payload or {}
        if "config" in payload and isinstance(payload.get("config"), dict):
            payload = payload["config"]

        clean, errors = self._validate_payload(payload)
        if errors:
            return {"ok": False, "errors": errors, "error": errors[0]}

        merged = self._read_config_file()
        merged.update(clean)

        written = self._write_config_file(merged)

        # Apply regardless: if the file write failed we still honour the change
        # for this run, and say so, rather than silently doing nothing.
        async with self._lock:
            self._apply_config(merged)

        self._log(
            f"WebUI updated config ({len(clean)} field(s), persisted={written}): "
            f"{', '.join(sorted(clean))}"
        )
        return {
            "ok": True,
            "persisted": written,
            "config": self._read_config_file(),
            "config_path": self._config_path(),
            "warnings": [] if written else ["配置未能写入磁盘，仅对本次运行生效"],
        }

    @register.api(method="POST", path="/toggle")
    async def api_toggle(self, payload: dict):
        """Enable/disable the plugin and persist the new state."""
        payload = payload or {}
        enabled = bool(payload.get("enabled", not self.enabled))

        merged = self._read_config_file()
        merged["enabled"] = enabled
        written = self._write_config_file(merged)

        async with self._lock:
            self._apply_config(merged)

        self._log(f"WebUI toggled enabled={enabled} (persisted={written})")
        return {
            "ok": True,
            "enabled": self.enabled,
            "persisted": written,
            "warnings": [] if written else ["配置未能写入磁盘，仅对本次运行生效"],
        }

    @register.widget(
        label={"zh": "草傻子专用", "en": "Caoshazi"},
        icon="DataLine",
        color="red",
        order=80,
        size="small",
    )
    async def widget_active_scenes(self) -> str:
        """Overview dashboard widget: number of live scenes."""
        if not self.enabled:
            return "off"
        return str(sum(1 for st in self.states.values() if st.active))
