import { generateText } from "ai";
import fs from "node:fs/promises";
import path from "node:path";
import { z } from "zod";

const ProviderMetadata = z.object({
  gateway: z.object({
    cost: z.string(),
  }),
});

const latestChanges = [
  `--- a/src/utils.ts
+++ b/src/utils.ts
@@ -0,0 +1,3 @@
+export function fullName(firstName: string, lastName?: string): string {
+  return lastName ? \`\${firstName} \${lastName}\` : firstName;
+}`,
];

const code = `--- a/src/greeting.ts
+++ b/src/greeting.ts
@@ -0,0 +1 @@
+func`;

const prompt = `Given recent changes:

${latestChanges.join("\n\n")}

What the most logical completion for the following code state would be?

${code}

Answer with the most likely completion in plain TypeScript, without any additional text or formatting.
`;

const models = [
  "openai/gpt-oss-20b",
  "openai/gpt-oss-120b",
  "openai/gpt-5-nano",
  "openai/gpt-5-mini",
  "google/gemini-2.0-flash-lite",
  "google/gemini-2.5-flash-lite",
  "google/gemini-3.1-flash-lite-preview",
  "meta/llama-3.1-8b",
  "xai/grok-4.1-fast-non-reasoning",
  "xai/grok-code-fast-1",
  "alibaba/qwen-3-14b",
  "alibaba/qwen-3-30b",
  "alibaba/qwen-3-32b",
  "alibaba/qwen-3-235b",
  "alibaba/qwen3.5-flash",
  "mistral/ministral-3b",
  "zai/glm-4.7-flashx",
];

interface Result {
  model: string;
  cost: string;
  timingSec: number;
  inputTokens: number | undefined;
  outputTokens: number | undefined;
}

const results: Result[] = [];

const RESULTS_DIR = "results";
const WIDTH = 80;

for (const model of models) {
  console.log(`\n=== ${model} `.padEnd(WIDTH, "="));

  const started = Date.now();
  const completion = await generateText({ model, prompt });
  const timing = Date.now() - started;

  const timingSec = parseFloat((timing / 1000).toFixed(2));
  const { inputTokens, outputTokens } = completion.totalUsage;
  const metadata = ProviderMetadata.parse(completion.providerMetadata);
  const { cost } = metadata.gateway;

  results.push({
    model,
    cost,
    timingSec,
    inputTokens,
    outputTokens,
  });

  console.log(
    `\n${timingSec}s · $${cost} · input ${inputTokens || "?"} · output ${outputTokens || "?"}\n`,
  );

  console.log("-".repeat(WIDTH));
  console.log(`${completion.output}`);
  console.log("=".repeat(WIDTH) + "\n");
}

await fs.mkdir(RESULTS_DIR, { recursive: true });
await fs.writeFile(
  path.resolve(RESULTS_DIR, String(Date.now())),
  JSON.stringify({ prompt, results }, null, 2),
);
