/**
 * Token sampling over a final-shard logits tensor. Runs on whichever peer holds
 * the last shard.
 */

export interface SampleOptions {
  temperature?: number;
  topP?: number;
  topK?: number;
}

/** Extract the logits row for the last position from a [1, seq, vocab] tensor. */
export function lastTokenLogits(data: Float32Array, dims: number[]): Float32Array {
  const seq = dims[1];
  const vocab = dims[2];
  const offset = (seq - 1) * vocab;
  return data.subarray(offset, offset + vocab);
}

function argmax(logits: Float32Array): number {
  let best = 0;
  let bestVal = -Infinity;
  for (let i = 0; i < logits.length; i++) {
    if (logits[i] > bestVal) {
      bestVal = logits[i];
      best = i;
    }
  }
  return best;
}

export function sampleToken(
  logits: Float32Array,
  opts: SampleOptions = {},
): number {
  const temperature = opts.temperature ?? 0;
  if (temperature <= 0) return argmax(logits);

  // Build (index, logit) candidates, optionally restricted by top-k.
  const topK = opts.topK && opts.topK > 0 ? opts.topK : logits.length;
  let indices = Array.from(logits.keys());
  indices.sort((a, b) => logits[b] - logits[a]);
  if (topK < indices.length) indices = indices.slice(0, topK);

  // Softmax over the candidates with temperature.
  const maxLogit = logits[indices[0]];
  const scaled = indices.map((i) => Math.exp((logits[i] - maxLogit) / temperature));
  const sum = scaled.reduce((a, b) => a + b, 0);
  const probs = scaled.map((p) => p / sum);

  // Nucleus (top-p) truncation.
  const topP = opts.topP ?? 1;
  let cutoff = indices.length;
  if (topP < 1) {
    let cumulative = 0;
    for (let i = 0; i < probs.length; i++) {
      cumulative += probs[i];
      if (cumulative >= topP) {
        cutoff = i + 1;
        break;
      }
    }
  }

  // Sample from the (renormalized) truncated distribution.
  let mass = 0;
  for (let i = 0; i < cutoff; i++) mass += probs[i];
  let r = Math.random() * mass;
  for (let i = 0; i < cutoff; i++) {
    r -= probs[i];
    if (r <= 0) return indices[i];
  }
  return indices[0];
}
