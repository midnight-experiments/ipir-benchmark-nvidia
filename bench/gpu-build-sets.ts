/** Rows of the six tables of the block budget's sets (`bench/budget.ts`): one column block of each changes per chain block. */
export const SET_ROWS: Record<string, number[]> = {
  today: [2048, 2048, 2048, 6144, 2048, 2048],
  "10m": [14336, 16384, 8192, 6144, 2048, 2048],
};
