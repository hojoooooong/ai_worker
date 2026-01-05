"use client";

import type { ReactNode } from "react";

interface StatusBadgeProps {
  status: boolean | string;
  label?: string;
  className?: string;
}

export default function StatusBadge({
  status,
  label,
  className = "",
}: StatusBadgeProps) {
  const isUp =
    typeof status === "boolean" ? status : status.toLowerCase() === "running";

  const baseClasses =
    "inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium";
  const statusClasses = isUp
    ? "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200"
    : "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200";

  const displayLabel = label || (isUp ? "Running" : "Stopped");

  return (
    <span className={`${baseClasses} ${statusClasses} ${className}`}>
      <span
        className={`w-1.5 h-1.5 rounded-full mr-1.5 ${
          isUp ? "bg-green-500" : "bg-red-500"
        }`}
      />
      {displayLabel}
    </span>
  );
}
