# Workload files

The workload is JSON Lines (`.jsonl`): one request per line. Each line must
contain either `messages` (OpenAI chat format) or a string `prompt`. Optional
`id` and `metadata` fields are carried into the per-request results.

Example schema (not a benchmark dataset):

```json
{"id":"case-001","messages":[{"role":"user","content":"..."}],"metadata":{"group":"..."}}
{"id":"case-002","prompt":"..."}
```

No workload is selected yet. Add an approved workload here when the benchmark
matrix and prompt set are decided. Keep the same file and order for each method
when a paired comparison is desired.
