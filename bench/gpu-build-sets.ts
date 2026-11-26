/**
 * The six tables of the block budget's sets (`bench/budget.ts`): one column block of each changes per chain block.
 *
 * `today` and `10m` are the sets every full run measures. `LADDER` holds the sizes between them: when the `10m` tables
 * do not fit in the GPU's memory with the blocks built on the GPU, `run-all.sh` goes down it and measures the first
 * size that fits, the largest tables that card can keep while it builds its blocks.
 *
 *   deno run bench/gpu-build-sets.ts      prints the ladder's set names, largest first
 */

/** The three Dust tables, the same at every number of shielded outputs. */
const DUST: [string, number, number][] = [["Dust groups", 6144, 28672], ["first coins", 2048, 2048], ["generation groups", 2048, 4096]];
const SHIELDED = ["read", "nullifier", "shielded groups"];

export interface BudgetTable {
  name: string;
  rows: number;
  cols: number;
}
const set = (shielded: [number, number][]): BudgetTable[] => [
  ...shielded.map(([rows, cols], i) => ({ name: SHIELDED[i]!, rows, cols })),
  ...DUST.map(([name, rows, cols]) => ({ name, rows, cols })),
];

/**
 * Sizes below 10 million shielded outputs, largest first: the read, nullifier and shielded-groups tables (rows x
 * columns) that the design's shape rule gives at that many outputs. The rule picks each table's shape on its own, so
 * memory does not fall at every step (9 million outputs take more than 10 million).
 */
export const LADDER: { set: string; outputs: number; tables: BudgetTable[] }[] = ([
  ["9.5m", 9.5e6, [[14336, 32768], [8192, 34816], [8192, 20480]]],
  ["9m", 9e6, [[12288, 36864], [8192, 34816], [6144, 24576]]],
  ["8.5m", 8.5e6, [[12288, 34816], [8192, 32768], [6144, 24576]]],
  ["8m", 8e6, [[12288, 32768], [8192, 30720], [8192, 16384]]],
  ["7.5m", 7.5e6, [[12288, 30720], [8192, 28672], [6144, 20480]]],
  ["7m", 7e6, [[12288, 28672], [8192, 26624], [6144, 20480]]],
  ["6.5m", 6.5e6, [[12288, 26624], [8192, 24576], [6144, 20480]]],
  ["6m", 6e6, [[12288, 24576], [8192, 22528], [6144, 16384]]],
  ["5.5m", 5.5e6, [[10240, 26624], [8192, 20480], [6144, 16384]]],
  ["5m", 5e6, [[10240, 24576], [8192, 18432], [4096, 20480]]],
  ["4m", 4e6, [[8192, 24576], [8192, 16384], [4096, 16384]]],
  ["3m", 3e6, [[8192, 18432], [8192, 12288], [4096, 12288]]],
  ["2m", 2e6, [[6144, 16384], [4096, 16384], [4096, 8192]]],
  ["1m", 1e6, [[4096, 12288], [4096, 8192], [2048, 8192]]],
] as [string, number, [number, number][]][]).map(([name, outputs, shielded]) => ({ set: name, outputs, tables: set(shielded) }));

/** Every set by name. */
export const SETS: Record<string, BudgetTable[]> = {
  today: set([[2048, 2048], [2048, 2048], [2048, 4096]]),
  "10m": set([[14336, 34816], [16384, 18432], [8192, 20480]]),
  ...Object.fromEntries(LADDER.map((x) => [x.set, x.tables])),
};
/** Shielded outputs of a set (null for `today`, whose tables are the starting shapes). */
export const SET_OUTPUTS: Record<string, number | null> = { today: null, "10m": 10e6, ...Object.fromEntries(LADDER.map((x) => [x.set, x.outputs])) };
/** Rows of the six tables of a set. */
export const SET_ROWS: Record<string, number[]> = Object.fromEntries(Object.entries(SETS).map(([k, v]) => [k, v.map((t) => t.rows)]));

if (import.meta.main) console.log(LADDER.map((x) => x.set).join(" "));
