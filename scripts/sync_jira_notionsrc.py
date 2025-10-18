#!/usr/bin/env python3
props = r.get("properties", {})
# Fallback last_edited_time if Updated missing
last_edited = r.get("last_edited_time")


key = norm_key(norm_people_or_text(props.get("Key")))
if not key:
continue # skip rows with no key


summary = norm_people_or_text(props.get("Summary") or props.get("Title"))
issue_type = enum_safe("Issue Type", norm_people_or_text(props.get("Issue Type")))
status = enum_safe("Status", norm_people_or_text(props.get("Status")))
priority = enum_safe("Priority", norm_people_or_text(props.get("Priority")))
reporter = norm_people_or_text(props.get("Reporter"))
assignee = norm_people_or_text(props.get("Assignee"))
sprint = norm_people_or_text(props.get("Sprint"))
epic = norm_people_or_text(props.get("Epic Link"))
parent = norm_people_or_text(props.get("Parent"))
labels = norm_labels(props.get("Labels"))
jira_url = norm_url(props.get("Jira URL"))


created, _ = norm_date_prop(props.get("Created"))
updated = extract_updated(props.get("Updated"), last_edited)
resolved, _ = norm_date_prop(props.get("Resolved"))
due, _ = norm_date_prop(props.get("Due date"))
story_points = norm_number(props.get("Story Points"))


mapped = {
"Key": key,
"Summary": summary,
"Issue Type": issue_type,
"Status": status,
"Priority": priority,
"Reporter": reporter,
"Assignee": assignee,
"Sprint": sprint,
"Epic Link": epic,
"Parent": parent,
"Labels": labels,
"Jira URL": jira_url,
"Created": created,
"Updated": updated,
"Resolved": resolved,
"Due date": due,
"Story Points": story_points,
}


# Track max watermark (string compare OK for ISO-8601)
if updated:
if self.new_watermark is None or updated > self.new_watermark:
self.new_watermark = updated


self.consolidated_upsert(mapped, src_db)


# batch pacing
if (self.stats["created"] + self.stats["updated"]) % self.batch_size == 0:
time.sleep(self.sleep_secs)


except Exception as e:
self.stats["errors"] += 1
print(f"ERROR key?={props.get('Key') if 'props' in locals() else '?'} reason={e}")


if not has_more:
break


# Update watermark if we progressed (not on full-scan with no updates)
if self.new_watermark:
self.set_watermark(self.new_watermark)


# Summary
print(json.dumps({
"fetched": self.stats["fetched"],
"created": self.stats["created"],
"updated": self.stats["updated"],
"skipped": self.stats["skipped"],
"errors": self.stats["errors"],
"new_watermark": self.new_watermark,
}, indent=2))




def main():
token = os.getenv("NOTION_TOKEN")
consolidated = os.getenv("CONSOLIDATED_DB_ID")
control_page = os.getenv("SYNC_CONTROL_PAGE_ID")
sources_csv = os.getenv("SOURCE_DB_IDS", "")
if not token or not consolidated or not control_page or not sources_csv:
print("Missing required env vars: NOTION_TOKEN, CONSOLIDATED_DB_ID, SYNC_CONTROL_PAGE_ID, SOURCE_DB_IDS", file=sys.stderr)
sys.exit(2)


sources = [s.strip() for s in sources_csv.split(",") if s.strip()]
client = Notion(token)
SyncRunner(client, consolidated, control_page, sources).run()




if __name__ == "__main__":
main()
