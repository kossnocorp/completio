import { generateText } from "ai";
import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { z } from "zod";

const ProviderMetadata = z.object({
  gateway: z.object({
    cost: z.string(),
  }),
});

const latestChanges = [
  `<change>
--- a/src/utils.ts
+++ b/src/utils.ts
@@ -0,0 +1,3 @@
+export function fullName(firstName: string, lastName?: string): string {
+  return lastName ? \`\${firstName} \${lastName}\` : firstName;
+}
</change>`,
];

const prompt = `
You're code completion agent helping a humand developer to write code.

Considering recent changes the human did (diffs in <change> meta tags) predict code completion for <current> file. Don't add any explanation, formatting, or meta tags. Answer with just code that have to be added right after <cursor>. Don't include code before or anfter.

Don't include changes that you would made before <cursor>. If recent changes include something relevant or that can be useful for the current file, utilize that.

Example (completing \`const helloW\`):

<example good>
orld = "Hello, world!";
</example

<example bad why="full code">
const helloWorld = "Hello, world!";
</example>

<example bad why="Markdown formatting">
\`\`\`ts
orld = "Hello, world!";
\`\`\`
</example>

<example bad why="extra text">
orld = "Hello, world!";

It is likely iconic "Hello, world!" const.
</example>

${latestChanges.join("\n\n")}

<current path="src/greeting.ts"
export func<cursor>
</current>
`;

const models = [
  "openai/gpt-oss-20b",
  "openai/gpt-oss-120b",
  "openai/o4-mini",
  "openai/gpt-4.1-nano",
  "openai/gpt-4.1-mini",
  "openai/gpt-4.1",
  "openai/gpt-4o",
  // "openai/gpt-5-nano",
  // "openai/gpt-5-mini",
  "openai/gpt-5.1-thinking",
  // "openai/gpt-5.1-instant",
  // "openai/gpt-5.1-codex",
  // "openai/gpt-5.1-codex-mini",
  "openai/gpt-5.2",
  // "anthropic/claude-3-haiku",
  "anthropic/claude-sonnet-4",
  // "anthropic/claude-haiku-4.5",
  "anthropic/claude-sonnet-4.5",
  // "google/gemini-2.0-flash",
  // "google/gemini-2.0-flash-lite",
  // "google/gemini-2.5-flash",
  // "google/gemini-2.5-flash-lite",
  "google/gemini-3-flash",
  "google/gemini-3.1-flash-lite-preview",
  "xai/grok-3-mini",
  "xai/grok-3",
  "xai/grok-4-fast-non-reasoning",
  "xai/grok-4.1-fast-non-reasoning",
  // "xai/grok-code-fast-1",
  // "nvidia/nemotron-nano-9b-v2",
  // "nvidia/nemotron-nano-12b-v2-vl",
  // "nvidia/nemotron-3-nano-30b-a3b",
  "mistral/devstral-small-2",
  "mistral/codestral",
  "mistral/devstral-2",
  "mistral/mistral-small",
  "mistral/mistral-nemo",
  "deepseek/deepseek-v3.1",
  "deepseek/deepseek-v3.1-terminus",
  "deepseek/deepseek-v3.2",
  "moonshotai/kimi-k2-turbo",
  "xiaomi/mimo-v2-flash",
];

type Result = ResultError | ResultSuccess;

interface ResultError {
  status: "error";
  model: string;
  timingSec: number;
  error: string;
}

interface ResultSuccess {
  status: "success";
  model: string;
  cost: string;
  timingSec: number;
  inputTokens: number | undefined;
  outputTokens: number | undefined;
  output: string;
}

const results: Result[] = [];

const EXAMPLE_DIR = path.dirname(fileURLToPath(import.meta.url));
const RESULTS_DIR = path.join(EXAMPLE_DIR, "results");
const WIDTH = 80;

for (const model of models) {
  console.log(`\n=== ${model} `.padEnd(WIDTH, "="));
  const started = Date.now();
  const calcTimingSec = () =>
    parseFloat(((Date.now() - started) / 1000).toFixed(2));

  try {
    const completion = await generateText({
      model,
      prompt,
      providerOptions: {
        gateway: {
          order: ["azure", "openai", "groq", "bedroock"],
        },

        openai: {
          reasoningEffort: "none",
          // TODO: Consider trying if it keeps explaining itself:
          // textVerbosity: "low",
        },
        antrhopic: {
          thinking: { type: "enabled", effort: "low" },
          speed: "fast",
        },
      },
    });

    const timingSec = calcTimingSec();

    const {
      output,
      totalUsage: { inputTokens, outputTokens },
    } = completion;
    const metadata = ProviderMetadata.parse(completion.providerMetadata);
    const { cost } = metadata.gateway;

    results.push({
      status: "success",
      model,
      cost,
      timingSec,
      inputTokens,
      outputTokens,
      output,
    });

    console.log(
      `\n${timingSec}s · $${cost} · input ${inputTokens || "?"} · output ${outputTokens || "?"}\n`,
    );

    console.log("-".repeat(WIDTH));
    console.log(`${output}`);
    console.log("=".repeat(WIDTH) + "\n");
  } catch (err) {
    const timingSec = calcTimingSec();

    results.push({
      status: "error",
      model,
      timingSec,
      error: String(err),
    });

    console.error(`\n!!! Failed to generate with ${model}: ${err}\n`);
  }
}

await fs.mkdir(RESULTS_DIR, { recursive: true });
await fs.writeFile(
  path.join(RESULTS_DIR, `${Date.now()}.json`),
  JSON.stringify({ prompt, results }, null, 2),
);
