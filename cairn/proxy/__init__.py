"""Cairn API proxy — strips Cairn artifacts from responses and injects
Cairn context into requests, keeping the model conversation clean and the
Anthropic prompt cache byte-exact. See docs/spec / plan for design."""
