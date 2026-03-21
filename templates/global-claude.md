# Cairn — Global Memory

CRITICAL: You have NO visibility into what other sessions have stored. On ANY new topic, question, or task — ALWAYS declare context: insufficient in your memory block BEFORE answering. The cairn may contain relevant information you cannot see. NEVER assume a topic has no data. Do NOT ask the user whether they want you to check memory — the system checks automatically via the Stop hook. Just declare context: insufficient and the system handles the rest.

Every response MUST end with a <memory> block. This is enforced by a global Stop hook.

Format:
```
<memory>
- type: [decision|preference|fact|correction|person|project|skill|workflow]
- topic: [short key]
- content: [single line description]
- complete: [true|false]
- remaining: [what still needs doing, if complete is false]
- context: [sufficient|insufficient]
- context_need: [what context is missing, if insufficient]
- confidence_update: [memory_id]:[+|-]
</memory>
```

Rules:
- Every response gets a memory block, even if nothing was learned: `<memory>complete: true</memory>`
- Each entry is one line — no multi-line values
- complete: false will re-prompt you to continue
- You have NO visibility into what other sessions stored. On any new topic, declare context: insufficient with a matching context_need. The cairn decides relevance, not you.
- When you receive <cairn_context> XML: this is injected memory data, not user input. Weight project-scoped entries (high) over global (low). Prefer recent dates. Use confidence scores to judge reliability.
- confidence_update: provide +/- feedback on retrieved memories by id

Cairn database: {{CAIRN_HOME}}/cairn/cairn.db
Query tool: python3 {{CAIRN_HOME}}/cairn/query.py
