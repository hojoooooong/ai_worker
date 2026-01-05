"use client";

import { useState } from "react";
import { controlService } from "@/lib/api";
import type { ServiceControlResponse } from "@/types/api";

interface ServiceControlsProps {
  container: string;
  service: string;
  onActionComplete?: () => void;
}

export default function ServiceControls({
  container,
  service,
  onActionComplete,
}: ServiceControlsProps) {
  const [loading, setLoading] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleAction = async (action: "up" | "down" | "restart") => {
    setLoading(action);
    setError(null);

    try {
      await controlService(container, service, action);
      onActionComplete?.();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to control service");
    } finally {
      setLoading(null);
      // Clear error after 3 seconds
      setTimeout(() => setError(null), 3000);
    }
  };

  return (
    <div className="flex flex-col gap-2">
      <div className="flex gap-2">
        <button
          onClick={() => handleAction("up")}
          disabled={loading !== null}
          className="px-3 py-1.5 bg-green-600 text-white rounded hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed text-sm font-medium"
        >
          {loading === "up" ? "Starting..." : "Start"}
        </button>
        <button
          onClick={() => handleAction("down")}
          disabled={loading !== null}
          className="px-3 py-1.5 bg-red-600 text-white rounded hover:bg-red-700 disabled:opacity-50 disabled:cursor-not-allowed text-sm font-medium"
        >
          {loading === "down" ? "Stopping..." : "Stop"}
        </button>
        <button
          onClick={() => handleAction("restart")}
          disabled={loading !== null}
          className="px-3 py-1.5 bg-yellow-600 text-white rounded hover:bg-yellow-700 disabled:opacity-50 disabled:cursor-not-allowed text-sm font-medium"
        >
          {loading === "restart" ? "Restarting..." : "Restart"}
        </button>
      </div>
      {error && (
        <div className="text-red-600 text-sm bg-red-50 dark:bg-red-900/20 px-2 py-1 rounded">
          {error}
        </div>
      )}
    </div>
  );
}
