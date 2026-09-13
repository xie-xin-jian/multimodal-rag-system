import fs from "node:fs";

const html = fs.readFileSync("templates/index.html", "utf8");
const scripts = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)];

if (scripts.length === 0) {
  throw new Error("no inline script found");
}

for (const [, source] of scripts) {
  new Function(source);
}

console.log("frontend syntax ok");
