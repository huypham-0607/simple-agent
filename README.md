# simple-agent

A simple exploratory agent. Its main functionality is to search for reference of certain patterns across files (function definition, call sites). Agent is sandboxed to avoid reading sensitive info.

## Project structure

```text
simple-agent/
├── src/simple_agent/
│   ├── agent.py            model setup, system prompt, agent construction
│   ├── main.py             CLI entry point
│   ├── tools.py            the three tools, and the sandbox check they share
│   └── __init__.py
│
├── notes/                  the only directory the agent can read
├── pyproject.toml          dependencies and the `simple-agent` entry point
├── .env                    GOOGLE_API_KEY; ignored by git
└── README.md
```

The agent is built with [deepagents](https://github.com/langchain-ai/deepagents) on top of LangGraph. The model is Gemini, called through `langchain-google-genai`.

`tools.py` holds all four tools. `agent.py` builds the agent from them and from the system prompt. `main.py` reads a question from the command line, runs the agent once, and prints the final answer.

## Tools

The agent has four tools:

- `list_dir` lists the direct children of one directory. It does not recurse.
- `grep` searches one file, or a whole directory when `recursive=True`. It returns matching lines without line numbers.
- `read_file` reads a line range from one file. It returns each line with its number, and stops at 8000 characters.

`grep` runs the system `grep` binary in a subprocess. Any `grep` on `PATH` will do, but the flags used are GNU syntax, so a GNU-compatible binary is required. This project has been run against ugrep 7.8.4.

## Sandbox

The agent can only reach files under `notes/`. Every path it passes to a tool goes through one check first. The check applies four rules:

1. A path is treated as a virtual path under `notes/`. `/models.py` and `models.py` both mean the same file. Host paths such as `/etc/passwd` cannot be written at all.
2. Paths containing `..` or `~` are rejected.
3. The path is fully resolved before it is checked, so a symlink that points out of `notes/` is rejected.
4. Files with a blocked name (`.env`, `.git`, `.ssh`, `id_rsa`, and similar) or a blocked suffix (`.pem`, `.key`, `.crt`) are rejected. `list_dir` also hides them, and `grep` skips them while searching.

A rejected path returns an error string to the model. It does not raise. The model reads the error and tries a different path.

This is a guard against the model, not process isolation. It stops the agent from asking for a sensitive file. It does not stop a bug in the tool code itself.

## Setup

Requirements:

- Python 3.14
- [uv](https://docs.astral.sh/uv/)
- a `grep` binary on `PATH`
- a Google Gemini API key

Install the project and its dependencies:

```bash
uv sync
```

Put the API key in `.env` at the repository root:

```bash
export GOOGLE_API_KEY=your-key-here
```

`.env` is ignored by git. The project does not read it automatically, so load it into the shell before running:

```bash
source .env
```

Finally, put the files you want to search into `notes/`. The directory is empty by default.

## Usage

Ask a question as a single argument:

```bash
uv run simple-agent "Where is get_device defined and what calls it?"
```

The agent lists the tree, searches it, reads the files it finds, and prints one answer. It cites each claim as `path:line`.

Requests are paced at one every five seconds, because the Gemini free tier limits requests per minute. A question that needs many searches will therefore take a while. The pacing is set in `agent.py`, together with the model name and the retry count.
