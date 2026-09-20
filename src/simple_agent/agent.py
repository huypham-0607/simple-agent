from langchain.chat_models import init_chat_model
from langchain_core.rate_limiters import InMemoryRateLimiter
from deepagents import create_deep_agent
from .tools import internet_search, grep, list_dir, read_file

# System prompt to steer the agent to trace symbols through the local file tree
navigator_instructions = r"""You answer questions about a local file tree — most often "where is X defined and what calls it" — using only `list_dir`, `grep` and `read_file` on virtual paths rooted at `/`, the top of the tree; nothing exists above it, and `internet_search` cannot see it at all.

Start with `list_dir("/")` to learn the tree's shape, then locate the definition by grepping for a definition *shape* rather than the bare name (`(def|class) +X\b` with `regexp=True`) and the callers by use shapes (`X *\(`, `\.X\b`, `(import|from).*X`).

Pass `recursive=True` whenever you grep a directory, and note that grep returns matching lines *without* line numbers — open every hit with `read_file`, which numbers each line, to find where the match actually sits.

A match is not a caller: definitions, imports, comments, docstrings, string literals and same-named locals all match a bare name, so read before you claim.

Report the definition site first and then each call site as `path:line`, citing only lines you actually opened, and say plainly when a name appears nowhere or when something is too ambiguous to resolve."""

rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.2,   # ~12/min, under typical free-tier RPM
    check_every_n_seconds=0.1,
    max_bucket_size=1,         # no bursting — the burst is what triggers 429s
)

model = init_chat_model(
    model="google_genai:gemini-3.1-flash-lite",
    thinking_level="medium",
    rate_limiter=rate_limiter,
    max_retries=3,
)

agent = create_deep_agent(
    model = model,
    tools = [grep, list_dir, read_file],
    system_prompt = navigator_instructions,
)
