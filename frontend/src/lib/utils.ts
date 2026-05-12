import { type ClassValue, clsx } from "clsx";

export function cn(...inputs: ClassValue[]) {
  return clsx(inputs);
}

export function formatPercent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}
