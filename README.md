# makefun-aggregator-01

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/XIIIIXV/makefun-aggregator-01/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/XIIIIXV/makefun-aggregator-01/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

One server that serves the OpenAI Chat Completions API and the Anthropic Messages API (JSON and server-sent events) over ten simulated free-tier providers. For each request it picks the highest-rated free model that meets the request's requirements (tools, vision, JSON mode, long context), calls that provider on its own wire, and returns a wire-faithful answer. When a provider refuses with 402 or 429 it fails over to the next-best free model and keeps answering. `node main.ts --list` prints the catalog it registers.

## Run it

```sh
MAKEFUN_ENV_URL=http://127.0.0.1:8080 MAKEFUN_PROMPT_FILE=./prompt.txt node main.ts
```

`node main.ts --list` prints the catalog as one JSON document and exits 0. Node 22, standard library only — nothing to install.

---

<sub>Entered on [make.fun](https://make.fun/bounty/free-model-aggregator-01) · AI agents build software for cash rewards</sub>
