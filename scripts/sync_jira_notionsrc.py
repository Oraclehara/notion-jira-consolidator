#!/usr/bin/env python3
# Notion-based Jira Consolidator (No Jira API)
# - relation-aware people and status resolution
# - robust Status writes (never blank-out; auto-seed/extend select options)
# - idempotent upserts with source-hash skip

import os, sys, time, json, hashlib, datetime as dt
from typing import Dict, Any, List, Optional, Tuple
import requests

NOTION_VERSION = os.getenv("NOTION_VERSION", "2022-06-28")
NOTION_API = "https://api.notion.com/v1"
SESSION = requests.Session()

# ---------- Notion API Wrapper ----------

class Notion:
    def __init__(self, token: str):
        self.h = {
            "Authorization": f"Bearer {token}",
            "Notion-Version": NOTION_VERSION,
            "Content-Type": "application/json",
        }

    def _req(self, method: str, path: str, **kw) -> Dict[str, Any]:
        url = f"{NOTION_API}{path}"
        backoff = 1.0
        for _ in range(8):
            try:
                resp = SESSION.request(method, url, headers=self.h, timeout=40, **kw)
            except requests.exceptions.RequestException:
                time.sleep(backoff)
                backoff = min(backoff * 2, 30)
                continue

            if resp.status_code == 429:
                retry_after = float(resp.headers.get("Retry-After", backoff))
                time.sleep(max(retry_after, backoff))
                backoff = min(backoff * 2, 30)
                continue

            if 500 <= resp.status_code < 600:
                time.sleep(backoff)
                backoff = min(backoff * 2, 30)
                continue

            if not resp.ok:
                raise RuntimeError(f"Notion API error {resp.status_code}: {resp.text}")

            return resp.json()
        raise RuntimeError("Exceeded retry attempts against Notion API")

    def db_query(self, db_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._req("POST", f"/databases/{db_id}/query", json=payload)

    def db_retrieve(self, db_id: str) -> Dict[str, Any]:
        return self._req("GET", f"/databases/{db_id}")

    def db_update(self, db_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._req("PATCH", f"/databases/{db_id}", json=payload)

    def page_retrieve(self, page_id: str) -> Dict[str, Any]:
        return self._req("GET", f"/pages/{page_id}")

    def page_update(self, page_id: str, properties: Dict[str, Any]) -> Dict[str, Any]:
        return self._req("PATCH", f"/pages/{page_id}", json={"properties": properties})

    def page_create(self, parent_db_id: str, properties: Dict[str, Any]) -> Dict[str, Any]:
        return self._req("POST", "/pages", json={"parent": {"database_id": parent_db_id}, "properties": properties})

# ---------- Normalization Helpers ----------

SAFE_ENUMS = {
    "Issue Type": {"default": "Other", "map": {}},
    "Status": {"default": "", "map": {}},   # keep empty default; we don't want to force "Unknown"
    "Priority": {"default": "None", "map": {}},
}

MAPPED_FIELDS = [
    "Key", "Summary", "Issue Type", "Status", "Priority",
    "Reporter", "Assignee", "Sprint", "Epic Link", "Parent", "Labels",
    "Jira URL", "Created", "Updated", "Resolved", "Due date", "Story Points"
]

def norm_key(v: Any) -> str:
    return str(v or "").strip().upper()

def norm_people_or_text(prop: Dict[str, Any]) -> str:
    if not prop:
        return ""
    t = prop.get("type")

    if t == "people":
        names = []
        for p in prop.get("people", []):
            nm = p.get("name") or (p.get("person") or {}).get("email") or ""
            if nm:
                names.append(nm)
        return ", ".join(names)

    if t == "rich_text":
        return "".join(rt.get("plain_text", "") for rt in prop.get("rich_text", [])).strip()

    if t == "title":
        return "".join(rt.get("plain_text", "") for rt in prop.get("title", [])).strip()

    if t == "select":
        sel = prop.get("select")
        return (sel or {}).get("name") or ""

    if t == "status":
        st = prop.get("status") or {}
        return (st.get("name") or "").strip()

    # formula
    if t == "formula":
        f = prop.get("formula") or {}
        ftype = f.get("type")
        if ftype == "string":
            return (f.get("string") or "").strip()
        if ftype == "number":
            n = f.get("number")
            return "" if n is None else str(n)
        if ftype == "boolean":
            return "true" if f.get("boolean") else "false"
        if ftype == "date":
            d = (f.get("date") or {}).get("start")
            return d or ""
        if ftype == "rich_text":
            return "".join(rt.get("plain_text", "") for rt in f.get("rich_text", [])).strip()
        return ""

    # rollup
    if t == "rollup":
        r = prop.get("rollup") or {}
        rtype = r.get("type")
        if rtype == "array":
            vals = []
            for it in r.get("array", []):
                it_type = it.get("type")
                if it_type == "status":
                    nm = ((it.get("status") or {}).get("name") or "").strip()
                    if nm:
                        vals.append(nm)
                elif it_type == "select":
                    nm = ((it.get("select") or {}).get("name") or "").strip()
                    if nm:
                        vals.append(nm)
                elif it_type in ("title", "rich_text", "people"):
                    vals.append(norm_people_or_text(it))
            return ", ".join(v for v in vals if v).strip()
        if rtype == "number":
            n = r.get("number")
            return "" if n is None else str(n)
        if rtype == "date":
            d = (r.get("date") or {}).get("start")
            return d or ""
        if rtype == "rich_text":
            return "".join(rt.get("plain_text", "") for rt in r.get("rich_text", [])).strip()
        return ""

    return ""

def norm_status(prop: Dict[str, Any]) -> str:
    if not prop:
        return ""
    t = prop.get("type")
    if t == "status":
        st = prop.get("status") or {}
        return (st.get("name") or "").strip()
    if t == "select":
        sel = prop.get("select") or {}
        return (sel.get("name") or "").strip()
    return norm_people_or_text(prop).strip()

def prop_first(props: Dict[str, Any], names: List[str]) -> Optional[Dict[str, Any]]:
    if not props:
        return None
    for n in names:
        if n in props and props[n]:
            return props[n]
    norm_map = {(k or "").strip().lower(): k for k in props.keys()}
    for n in names:
        k = norm_map.get((n or "").strip().lower())
        if k and props.get(k):
            return props[k]
    wanted = [s.strip().lower() for s in names if s]
    for k in props.keys():
        nk = (k or "").strip().lower()
        if any(w in nk for w in wanted) and props.get(k):
            return props[k]
    return None

def norm_labels(prop: Dict[str, Any]) -> str:
    items: List[str] = []
    if not prop:
        return ""
    t = prop.get("type")
    if t == "multi_select":
        for it in prop.get("multi_select", []):
            nm = (it or {}).get("name")
            if nm:
                items.append(nm)
    elif t in ("rich_text", "title"):
        txt = norm_people_or_text(prop)
        if txt:
            rough = []
            for chunk in txt.split(";"):
                rough += chunk.split(",")
            for r in rough:
                nm = r.strip()
                if nm:
                    items.append(nm)
    items = sorted(set(x.strip().lower() for x in items if x and x.strip()))
    return ", ".join(items)

def norm_date_prop(prop: Dict[str, Any]) -> Tuple[Optional[str], bool]:
    if not prop or prop.get("type") != "date":
        return (None, False)
    d = prop.get("date") or {}
    start = d.get("start")
    if not start:
        return (None, False)
    return start, ("T" in start)

def norm_number(prop: Dict[str, Any]) -> Optional[float]:
    if not prop or prop.get("type") != "number":
        return None
    return prop.get("number")

def norm_url(prop: Dict[str, Any]) -> str:
    if not prop:
        return ""
    if prop.get("type") == "url":
        return prop.get("url") or ""
    return norm_people_or_text(prop)

def enum_safe(name: str, value: str) -> str:
    conf = SAFE_ENUMS.get(name)
    if not conf:
        return value
    m = conf.get("map", {})
    return m.get(value, value or conf.get("default", ""))

def compute_source_hash(mapped: Dict[str, Any]) -> str:
    order = MAPPED_FIELDS
    s = "\u241f".join(str(mapped.get(k, "")) for k in order)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def extract_updated(prop: Dict[str, Any], fallback_last_edited_time: Optional[str]) -> Optional[str]:
    val, _ = norm_date_prop(prop)
    return val or fallback_last_edited_time

# ---------- Notion Property Builders ----------

def rich(v: str) -> Dict[str, Any]:
    return {"rich_text": [{"type": "text", "text": {"content": v[:2000]}}]} if v else {"rich_text": []}

def title(v: str) -> Dict[str, Any]:
    return {"title": [{"type": "text", "text": {"content": v[:2000]}}]} if v else {"title": []}

def sel(v: str) -> Dict[str, Any]:
    return {"select": {"name": v}} if v else {"select": None}

def status_prop(v: Optional[str]) -> Dict[str, Any]:
    return {"status": {"name": v}} if v else {"status": None}

def url(v: str) -> Dict[str, Any]:
    return {"url": v or None}

def num(v: Optional[float]) -> Dict[str, Any]:
    return {"number": v}

def date(v: Optional[str]) -> Dict[str, Any]:
    return {"date": {"start": v}} if v else {"date": None}

def ms(csv_text: Optional[str]) -> Dict[str, Any]:
    vals = []
    if csv_text:
        for part in csv_text.split(","):
            name = part.strip()
            if name:
                vals.append({"name": name})
    return {"multi_select": vals}

# ---------- Debug helpers ----------

def _debug_prop_names(props: Dict[str, Any], key_label: str):
    if os.getenv("DEBUG", "") != "1":
        return
    try:
        keys = list(props.keys())
        print(f"DEBUG[{key_label}] props keys: {keys[:40]}{'...' if len(keys)>40 else ''}")
        want = ["reporter", "assignee", "related jira reporter", "related jira assignee"]
        for k in keys:
            lk = k.lower()
            if any(w in lk for w in want):
                v = props.get(k)
                t = v.get("type") if isinstance(v, dict) else type(v).__name__
                head = str(v)[:200]
                print(f"DEBUG[{key_label}] candidate '{k}' type={t} value-head={head}")
    except Exception as e:
        print("DEBUG error while printing prop names:", e)

def _dbg_status(msg: str):
    if os.getenv("DEBUG_STATUS", "") == "1":
        print(msg)

# ---------- Consolidated Status Schema Helpers ----------

def _safe_lower(s: Optional[str]) -> str:
    return (s or "").strip().lower()

def _fetch_consolidated_status_spec(notion: Notion, db_id: str) -> Tuple[str, List[str]]:
    try:
        db = notion.db_retrieve(db_id)
        prop = (db.get("properties") or {}).get("Status") or {}
        ptype = prop.get("type") or ""
        options: List[str] = []
        if ptype == "status":
            options = [o.get("name", "") for o in (prop.get("status") or {}).get("options", [])]
        elif ptype == "select":
            options = [o.get("name", "") for o in (prop.get("select") or {}).get("options", [])]
        return ptype, options
    except Exception as e:
        print(f"WARNING: failed to read Consolidated.Status schema: {e}")
        return "", []

def _map_status_to_existing(name: str, existing: List[str]) -> Optional[str]:
    if not name:
        return None
    name_norm = _safe_lower(name)
    for opt in existing:
        if _safe_lower(opt) == name_norm:
            return opt
    ALIASES = {
        "to do": ["todo", "to-do", "to_do", "to  do"],
        "in progress": ["doing", "wip", "in-progress", "in  progress"],
        "ready for testing": ["qa ready", "ready for qa", "ready-for-testing"],
        "in review": ["code review", "review"],
        "blocked": ["on hold", "on-hold"],
        "done": ["completed", "resolved", "closed"],
        "not started": ["backlog", "new"],
        "closed": ["closed"],
        "open": ["open"],
    }
    for canonical, variants in ALIASES.items():
        if name_norm == canonical or name_norm in variants:
            for opt in existing:
                if _safe_lower(opt) in ([canonical] + variants):
                    return opt
    parts = set(name_norm.split())
    for opt in existing:
        if parts & set(_safe_lower(opt).split()):
            return opt
    return None

# ---------- Dynamic Status Option Helpers ----------

CANONICAL_STATUS_OPTIONS = [
    "TO DO", "In Progress", "Done", "Open", "Ready for Testing",
    "In Review", "Blocked", "In Analysis", "Completed", "Closed",
]

def ensure_select_option(notion: Notion, db_id: str, property_name: str, option_name: str):
    if not option_name:
        return option_name
    try:
        db = notion.db_retrieve(db_id)
        props = db.get("properties", {})
        prop = props.get(property_name, {})
        if not prop:
            return option_name
        if prop.get("type") == "select":
            existing = [o["name"] for o in prop["select"].get("options", [])]
            if option_name not in existing:
                notion.db_update(db_id, {
                    "properties": {
                        property_name: {
                            "select": {"options": existing and [{"name": e} for e in existing] or []}
                        }
                    }
                })  # no-op refresh to be safe
                notion.db_update(db_id, {
                    "properties": {
                        property_name: {
                            "select": {
                                "options": [{"name": e} for e in existing] + [{"name": option_name}]
                            }
                        }
                    }
                })
                print(f"INFO: Added new select option '{option_name}' to '{property_name}'")
    except Exception as e:
        print(f"WARNING: Failed to ensure option '{option_name}' for '{property_name}': {e}")
    return option_name

def seed_status_options_if_needed(notion: Notion, db_id: str):
    """
    If the Consolidated 'Status' is a select with very few options, seed a canonical set.
    We never remove existing options, only add missing ones.
    """
    try:
        db = notion.db_retrieve(db_id)
        prop = (db.get("properties") or {}).get("Status") or {}
        if prop.get("type") != "select":
            return
        existing = [o.get("name", "") for o in (prop.get("select") or {}).get("options", [])]
        # heuristic: seed if options list looks suspiciously tiny
        if len(existing) < 3:
            to_add = [o for o in CANONICAL_STATUS_OPTIONS if o not in existing]
            if to_add:
                new_opts = [{"name": o} for o in existing + to_add]
                notion.db_update(db_id, {"properties": {"Status": {"select": {"options": new_opts}}}})
                print(f"INFO: Seeded {len(to_add)} Status options: {', '.join(to_add)}")
    except Exception as e:
        print(f"WARNING: Unable to seed Status options: {e}")

def _is_validation_error(e: Exception) -> bool:
    s = str(e)
    return "validation_error" in s or "Invalid status option" in s or "property type" in s

def _build_status_payloads(
    raw_status: str,
    consolidated_type: str,
    consolidated_options: List[str],
    notion: Notion,
    consolidated_db: str
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    raw = (raw_status or "").strip()
    if not raw:
        return candidates

    def _select_payload():
        ensured = ensure_select_option(notion, consolidated_db, "Status", raw)
        return {"select": {"name": ensured}}

    def _status_payload():
        mapped = _map_status_to_existing(raw, consolidated_options) if consolidated_options else raw
        return {"status": {"name": mapped}}

    if consolidated_type == "select":
        candidates = [_select_payload(), _status_payload()]
    elif consolidated_type == "status":
        candidates = [_status_payload(), _select_payload()]
    else:
        candidates = [_select_payload(), _status_payload()]
    return candidates

# ---------- Sync Logic ----------

class SyncRunner:
    def __init__(self, notion: Notion, consolidated_db: str, sync_control_db: str, source_db_ids: List[str]):
        self.consolidated_status_type: str = ""
        self.consolidated_status_options: List[str] = []
        self.notion = notion
        self.consolidated_db = consolidated_db
        self.sync_control_db = sync_control_db
        self.source_db_ids = source_db_ids
        self.max_pages = int(os.getenv("MAX_PAGES_PER_RUN", "50"))
        self.page_size = int(os.getenv("PAGE_SIZE", "100"))
        self.batch_size = int(os.getenv("BATCH_SIZE", "10"))
        self.sleep_secs = float(os.getenv("SLEEP_SECS", "0.4"))
        self.stats = {"fetched": 0, "created": 0, "updated": 0, "skipped": 0, "errors": 0}
        self.new_watermark: Optional[str] = None

    # --- Relation resolution helpers ---

    def _page_title_text(self, page: Dict[str, Any]) -> str:
        props = page.get("properties", {}) or {}
        for _, p in props.items():
            if isinstance(p, dict) and p.get("type") == "title":
                txt = "".join(rt.get("plain_text", "") for rt in p.get("title", []))
                if txt.strip():
                    return txt.strip()
        for candidate in ["Name", "Full Name", "Display Name", "Title"]:
            p = props.get(candidate)
            if isinstance(p, dict):
                t = p.get("type")
                if t == "rich_text":
                    txt = "".join(rt.get("plain_text", "") for rt in p.get("rich_text", []))
                    if txt.strip():
                        return txt.strip()
                if t == "people":
                    people = p.get("people", [])
                    if people:
                        nm = people[0].get("name") or (people[0].get("person") or {}).get("email") or ""
                        if nm:
                            return nm
        return ""

    def _relation_to_names(self, prop: Dict[str, Any], max_items: int = 3) -> str:
        if not prop or prop.get("type") != "relation":
            return ""
        rel = prop.get("relation", []) or []
        out: List[str] = []
        for it in rel[:max_items]:
            pid = it.get("id")
            if not pid:
                continue
            try:
                pg = self.notion.page_retrieve(pid)
                nm = self._page_title_text(pg)
                if nm:
                    out.append(nm)
            except Exception as e:
                print(f"DEBUG relation fetch error for {pid}: {e}")
        return ", ".join(out)

    def _extract_person_like(self, props: Dict[str, Any], name_candidates: List[str]) -> str:
        p = prop_first(props, name_candidates)
        if not p:
            return ""
        if p.get("type") == "relation":
            v = self._relation_to_names(p)
            if v:
                return v
        return norm_people_or_text(p)

    # --- Control DB (watermark) ---

    def _get_control_row_id_and_value(self) -> Tuple[Optional[str], Optional[str], int]:
        res = self.notion.db_query(self.sync_control_db, {"page_size": 1})
        rows = res.get("results", [])
        if not rows:
            # Create the control row with both fields initialized
            created = self.notion.page_create(
                self.sync_control_db,
                {
                    "Name": title("Main Control"),
                    "Last Watermark (Date)": date(None),
                    "Last Source Index": {"number": 0},
                },
            )
            rows = [created]
    
        row = rows[0]
        props = row.get("properties", {}) or {}
    
        # Watermark
        lw = props.get("Last Watermark (Date)")
        iso_val = None
        if lw and lw.get("type") == "date" and lw.get("date"):
            iso_val = lw["date"].get("start")
    
        # Source index (default 0 if missing)
        lsi_prop = props.get("Last Source Index") or {}
        start_idx = 0
        if lsi_prop.get("type") == "number":
            n = lsi_prop.get("number")
            if isinstance(n, (int, float)) and n is not None:
                start_idx = int(n) if n >= 0 else 0
    
        return row["id"], iso_val, start_idx

    def _set_control_source_index(self, row_id: str, idx: int):
        # Persist the index we want the next run to start from
        self.notion.page_update(row_id, {"Last Source Index": {"number": idx}})
    # --- Consolidated DB lookup ---

    def consolidated_find_by_key(self, key: str) -> Optional[Dict[str, Any]]:
        payload = {
            "filter": {"property": "Key", "title": {"equals": key}},
            "page_size": 1,
        }
        res = self.notion.db_query(self.consolidated_db, payload)
        results = res.get("results", [])
        return results[0] if results else None

    def _create_or_update_with_status_fallback(
        self,
        current: Optional[Dict[str, Any]],
        base_props: Dict[str, Any],
        status_payload_candidates: List[Dict[str, Any]],
    ) -> None:
        """
        Attempts to create/update the page, trying Status payload candidates in order.
        If a 400 validation error occurs, it retries with the next candidate.
        """
        # No Status updates at all → keep existing Status
        if not status_payload_candidates:
            if current is None:
                self.notion.page_create(self.consolidated_db, base_props)
                self.stats["created"] += 1
            else:
                self.notion.page_update(current["id"], base_props)
                self.stats["updated"] += 1
            return

        last_err = None
        for cand in status_payload_candidates:
            props = dict(base_props)
            props["Status"] = cand
            try:
                if current is None:
                    self.notion.page_create(self.consolidated_db, props)
                    self.stats["created"] += 1
                else:
                    self.notion.page_update(current["id"], props)
                    self.stats["updated"] += 1
                return
            except Exception as e:
                last_err = e
                if _is_validation_error(e):
                    continue
                raise

        print(f"INFO: All Status payload variants failed validation; saving without Status. reason={last_err}")
        if current is None:
            self.notion.page_create(self.consolidated_db, base_props)
            self.stats["created"] += 1
        else:
            self.notion.page_update(current["id"], base_props)
            self.stats["updated"] += 1

    # --- Upsert with robust Status handling ---

    def consolidated_upsert(self, mapped: Dict[str, Any], src_db_id: str) -> None:
        raw_status = (mapped.get("Status") or "").strip()

        # Base props (exclude Status; we add via candidates below to avoid blanking)
        src_hash = compute_source_hash(mapped)
        base_props = {
            "Key": title(mapped["Key"]),
            "Summary": rich(mapped.get("Summary", "")),
            "Issue Type": sel(mapped.get("Issue Type", "")),
            "Priority": sel(mapped.get("Priority", "")),
            "Reporter": rich(mapped.get("Reporter", "")),
            "Assignee": rich(mapped.get("Assignee", "")),
            "Sprint": rich(mapped.get("Sprint", "")),
            "Epic Link": rich(mapped.get("Epic Link", "")),
            "Parent": rich(mapped.get("Parent", "")),
            "Labels": ms(mapped.get("Labels", "")),
            "Jira URL": url(mapped.get("Jira URL", "")),
            "Created": date(mapped.get("Created")),
            "Updated": date(mapped.get("Updated")),
            "Resolved": date(mapped.get("Resolved")),
            "Due date": date(mapped.get("Due date")),
            "Story Points": num(mapped.get("Story Points")),
            "Source Hash": rich(src_hash),
            "Source Database": rich(src_db_id),
        }

        # Skip unchanged (unless forced)
        current = self.consolidated_find_by_key(mapped["Key"])
        if current is not None and os.getenv("FORCE_UPDATE_ALL", "") != "1":
            cur_hash = ""
            try:
                prop = current.get("properties", {}).get("Source Hash", {})
                cur_hash = "".join(t.get("plain_text", "") for t in prop.get("rich_text", []))
            except Exception:
                cur_hash = ""
            if cur_hash == src_hash:
                self.stats["skipped"] += 1
                return

        # Build candidate Status payloads based on schema
        status_candidates = _build_status_payloads(
            raw_status,
            self.consolidated_status_type,
            self.consolidated_status_options,
            self.notion,
            self.consolidated_db,
        )

        # Try to upsert
        self._create_or_update_with_status_fallback(current, base_props, status_candidates)

    # ---------- Status extraction ----------

    def _extract_status_value(self, props: Dict[str, Any]) -> str:
        """
        Resolve a display-ready status from many shapes:
        - direct: status/select/rich_text/title
        - relation: fetch related page → prefer its Status/select → fallback to related page title
        - rollup: unpack array → handle select/status/relation items → fallback to strings
        - formula: string
        """
        src_status_prop = props.get("Status") or prop_first(
            props,
            [
                "Status",
                "Status (Jira)",
                "Issue Status",
                "State",
                "Status name",
                "Status Name",
                "Jira Status",
                "Current status",
                "Workflow State",
            ],
        )
        if not src_status_prop:
            return ""

        t = src_status_prop.get("type")

        if t == "relation":
            rel = (src_status_prop.get("relation") or [])
            for it in rel:
                pid = it.get("id")
                if not pid:
                    continue
                try:
                    pg = self.notion.page_retrieve(pid)
                    p2 = (pg or {}).get("properties", {}) or {}
                    rp = p2.get("Status") or prop_first(
                        p2,
                        [
                            "Status",
                            "Status (Jira)",
                            "Issue Status",
                            "State",
                            "Status name",
                            "Status Name",
                            "Jira Status",
                            "Current status",
                            "Workflow State",
                        ],
                    )
                    val = norm_status(rp)
                    if val:
                        _dbg_status(f"DEBUG_STATUS relation→page {pid} status_prop='{val}'")
                        return val
                    title_txt = self._page_title_text(pg)
                    if title_txt:
                        _dbg_status(f"DEBUG_STATUS relation→page {pid} title_fallback='{title_txt}'")
                        return title_txt
                except Exception as e:
                    _dbg_status(f"DEBUG_STATUS relation deref fail {pid}: {e}")
            return ""

        if t == "rollup":
            rl = src_status_prop.get("rollup") or {}
            rtype = rl.get("type")
            if rtype == "array":
                arr = rl.get("array") or []
                for item in arr:
                    if not isinstance(item, dict):
                        continue
                    itype = item.get("type")
                    if itype in ("select", "status", "rich_text", "title"):
                        val = norm_status(item) if itype in ("status", "select") else norm_people_or_text(item)
                        if val:
                            _dbg_status(f"DEBUG_STATUS rollup array -> '{val}'")
                            return val
                    elif itype == "relation":
                        rel2 = (item.get("relation") or [])
                        for rel_it in rel2:
                            pid = rel_it.get("id")
                            if not pid:
                                continue
                            try:
                                pg = self.notion.page_retrieve(pid)
                                p2 = (pg or {}).get("properties", {}) or {}
                                rp = p2.get("Status") or prop_first(
                                    p2,
                                    [
                                        "Status",
                                        "Status (Jira)",
                                        "Issue Status",
                                        "State",
                                        "Status name",
                                        "Status Name",
                                        "Jira Status",
                                        "Current status",
                                        "Workflow State",
                                    ],
                                )
                                val = norm_status(rp) or self._page_title_text(pg)
                                if val:
                                    _dbg_status(f"DEBUG_STATUS rollup relation→page {pid} '{val}'")
                                    return val
                            except Exception as e:
                                _dbg_status(f"DEBUG_STATUS rollup relation deref fail {pid}: {e}")
            elif rtype == "string":
                s = rl.get("string") or ""
                if s.strip():
                    return s.strip()
            elif rtype == "number":
                n = rl.get("number")
                if n is not None:
                    return str(n)
            return ""

        if t == "formula":
            f = src_status_prop.get("formula") or {}
            if f.get("type") == "string":
                return (f.get("string") or "").strip()
            return ""

        return norm_status(src_status_prop)

    # --- Run ---

    def _should_full_scan(self) -> bool:
        if os.getenv("FORCE_FULL_SCAN"):
            return True
        return dt.datetime.utcnow().weekday() == int(os.getenv("FULL_SCAN_WEEKDAY", "6"))

    def run(self):
        control_row_id, watermark, start_idx = self._get_control_row_id_and_value()
    
        # Determine whether to full scan; full scan ignores watermark (but still honors round-robin order)
        if self._should_full_scan():
            watermark = None
    
        # Ensure Consolidated.Status has healthy options before we start
        seed_status_options_if_needed(self.notion, self.consolidated_db)
    
        # Detect schema (and cache options)
        self.consolidated_status_type, self.consolidated_status_options = _fetch_consolidated_status_spec(
            self.notion, self.consolidated_db
        )
        if self.consolidated_status_type:
            print(
                f"INFO Consolidated.Status type: {self.consolidated_status_type} "
                f"({len(self.consolidated_status_options)} options)"
            )
        else:
            print("WARNING: Could not determine Consolidated.Status property type")
    
        # --- Round-robin rotation ---
        total_sources = len(self.source_db_ids)
        if total_sources == 0:
            print("INFO: No source databases provided.")
            return
        if start_idx < 0 or start_idx >= total_sources:
            start_idx = 0  # sanitize
    
        # We'll iterate sources in rotated order starting from start_idx
        # and persist "next start" after each source finishes.
        for offset in range(total_sources):
            src_db = self.source_db_ids[(start_idx + offset) % total_sources]
    
            next_start = (start_idx + offset + 1) % total_sources  # where the *next run* should begin
    
            next_cursor = None
            while True:
                if self.stats["fetched"] >= self.max_pages:
                    break
    
                payload: Dict[str, Any] = {
                    "page_size": self.page_size,
                    "sorts": [
                        {"property": "Updated", "direction": "descending"},
                        {"timestamp": "last_edited_time", "direction": "descending"},
                    ],
                }
    
                if watermark:
                    payload["filter"] = {"property": "Updated", "date": {"on_or_after": watermark}}
    
                if next_cursor:
                    payload["start_cursor"] = next_cursor
    
                data = self.notion.db_query(src_db, payload)
                results = data.get("results", [])
                next_cursor = data.get("next_cursor")
                has_more = data.get("has_more", False)
    
                for r in results:
                    self.stats["fetched"] += 1
                    try:
                        props = r.get("properties", {})
                        last_edited = r.get("last_edited_time")
    
                        if self.stats["fetched"] <= 3 and os.getenv("DEBUG", "") == "1":
                            _debug_prop_names(props, key_label=f"{src_db}:{self.stats['fetched']}")
    
                        key = norm_key(norm_people_or_text(props.get("Key")))
                        if not key:
                            continue
    
                        reporter_val = self._extract_person_like(
                            props,
                            [
                                "Related Jira Reporter",
                                "Reporter",
                                "Reporter (Jira)",
                                "Reported By",
                                "Creator",
                                "Author",
                                "Reporter Name",
                            ],
                        )
                        assignee_val = self._extract_person_like(
                            props,
                            [
                                "Related Jira Assignee",
                                "Assignee",
                                "Assignee (Jira)",
                                "Assigned To",
                                "Owner",
                                "Handler",
                                "Assignee Name",
                            ],
                        )
    
                        status_val = self._extract_status_value(props)
                        if os.getenv("DEBUG_STATUS", "") == "1" and self.stats["fetched"] <= 10:
                            st_prop = props.get("Status") or prop_first(
                                props,
                                [
                                    "Status",
                                    "Status (Jira)",
                                    "Issue Status",
                                    "State",
                                    "Status name",
                                    "Status Name",
                                    "Jira Status",
                                    "Current status",
                                    "Workflow State",
                                ],
                            )
                            st_type = (st_prop or {}).get("type") if isinstance(st_prop, dict) else type(st_prop).__name__
                            _dbg_status(
                                f"DEBUG_STATUS src_key={key} src_status_type={st_type} resolved='{status_val}'"
                            )
    
                        mapped = {
                            "Key": key,
                            "Summary": norm_people_or_text(
                                prop_first(props, ["Summary", "Title", "Issue summary", "Name"])
                            ),
                            "Issue Type": enum_safe(
                                "Issue Type",
                                norm_people_or_text(prop_first(props, ["Issue Type", "Type", "Issue type"])),
                            ),
                            "Status": enum_safe("Status", status_val),
                            "Priority": enum_safe(
                                "Priority", norm_people_or_text(prop_first(props, ["Priority", "Issue Priority"]))
                            ),
                            "Reporter": reporter_val,
                            "Assignee": assignee_val,
                            "Sprint": norm_people_or_text(prop_first(props, ["Sprint", "Iteration", "Milestone"])),
                            "Epic Link": norm_people_or_text(prop_first(props, ["Epic Link", "Epic", "Parent Epic"])),
                            "Parent": norm_people_or_text(prop_first(props, ["Parent", "Parent Issue"])),
                            "Labels": norm_labels(prop_first(props, ["Labels", "Label", "Tags"])),
                            "Jira URL": norm_url(prop_first(props, ["Jira URL", "URL", "Link"])),
                            "Created": norm_date_prop(
                                prop_first(props, ["Created", "Created time", "Created Time"])
                            )[0],
                            "Updated": extract_updated(
                                prop_first(props, ["Updated", "Updated time", "Last Updated"]), last_edited
                            ),
                            "Resolved": norm_date_prop(prop_first(props, ["Resolved", "Resolution date"]))[0],
                            "Due date": norm_date_prop(prop_first(props, ["Due Date", "Due date", "Due"]))[0],
                            "Story Points": norm_number(prop_first(props, ["Story Points", "Points", "SP"])),
                        }
    
                        if os.getenv("DEBUG_STATUS", "") == "1" and self.stats["fetched"] <= 10:
                            _dbg_status(f"DEBUG_STATUS src_key={key} final_mapped_status='{mapped['Status']}'")
    
                        if mapped["Updated"]:
                            if self.new_watermark is None or mapped["Updated"] > self.new_watermark:
                                self.new_watermark = mapped["Updated"]
    
                        self.consolidated_upsert(mapped, src_db)
    
                        if (self.stats["created"] + self.stats["updated"]) % self.batch_size == 0:
                            time.sleep(self.sleep_secs)
    
                    except Exception as e:
                        self.stats["errors"] += 1
                        print(f"ERROR key={props.get('Key') if 'props' in locals() else '?'} reason={e}")
    
                if not has_more:
                    break
    
            # We completed this source; advance the round-robin pointer for the *next run*
            if control_row_id:
                try:
                    self._set_control_source_index(control_row_id, next_start)
                except Exception as e:
                    print(f"WARNING: failed to persist Last Source Index: {e}")
    
        # After all sources, commit watermark (if any)
        if self.new_watermark and control_row_id:
            self._set_control_value(control_row_id, self.new_watermark)
    
        print(
            json.dumps(
                {
                    "fetched": self.stats["fetched"],
                    "created": self.stats["created"],
                    "updated": self.stats["updated"],
                    "skipped": self.stats["skipped"],
                    "errors": self.stats["errors"],
                    "new_watermark": self.new_watermark,
                },
                indent=2,
            )
        )

# ---------- Entry Point ----------

def main():
    token = os.getenv("NOTION_TOKEN")
    consolidated = os.getenv("CONSOLIDATED_DB_ID")
    control_db = os.getenv("SYNC_CONTROL_PAGE_ID")
    sources_csv = os.getenv("SOURCE_DB_IDS", "")
    if not token or not consolidated or not control_db or not sources_csv:
        print(
            "Missing required env vars: NOTION_TOKEN, CONSOLIDATED_DB_ID, SYNC_CONTROL_PAGE_ID, SOURCE_DB_IDS",
            file=sys.stderr,
        )
        sys.exit(2)
    sources = [s.strip() for s in sources_csv.split(",") if s.strip()]
    client = Notion(token)
    SyncRunner(client, consolidated, control_db, sources).run()

if __name__ == "__main__":
    main()
