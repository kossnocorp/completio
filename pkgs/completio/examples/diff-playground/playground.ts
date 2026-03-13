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
  `<change>
--- a/src/utils.ts
+++ b/src/utils.ts
@@ -0,0 +1,3 @@
+export function fullName(firstName: string, lastName?: string): string {
+  return lastName ? \`\${firstName} \${lastName}\` : firstName;
+}
</change>`,
];

const prompt = `Given recent changes:

<changes>
${latestChanges.join("\n\n")}
</changes>

Complete with the most likely code continuation starting from <cursor> without any additional text or formatting, just plain code completion:

<code path="src/greeting.ts"
export func<cursor>
</code>
`;

const models = [
  "openai/gpt-oss-20b",
  "openai/gpt-oss-120b",
  "openai/gpt-5-nano",
  "openai/gpt-5-mini",
  "openai/gpt-5.1-codex-mini",
  "google/gemini-2.0-flash-lite",
  "google/gemini-2.5-flash-lite",
  "google/gemini-3.1-flash-lite-preview",
  "meta/llama-3.1-8b",
  "xai/grok-4.1-fast-non-reasoning",
  "alibaba/qwen-3-14b",
  "alibaba/qwen-3-30b",
  "alibaba/qwen-3-32b",
  "mistral/ministral-3b",
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
  path.resolve(RESULTS_DIR, `${Date.now()}.json`),
  JSON.stringify({ prompt, results }, null, 2),
);
