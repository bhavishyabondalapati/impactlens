# Decisions

Judgment calls made while resuming work. Design choices are explained in README.md and docs/.

1. **Committed `scripts/set_anthropic_key.sh` (2026-11-10).** It was left untracked by an earlier
   session. It holds no key: it reads one at a hidden prompt and writes it into a `.env` atomically
   with mode 600. Before committing it was run against a scratch `.env` with a fake `sk-ant-` value
   (key replaced in place, other lines kept, a non-Anthropic value rejected with nothing changed).
2. **No AWS work this session.** The local Terraform state has 0 resources, so the stack is down.
   `make deploy` creates paid resources and is left for the owner.
