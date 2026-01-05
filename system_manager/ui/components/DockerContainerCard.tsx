"use client";

import { useState } from "react";
import StatusBadge from "./StatusBadge";
import {
  controlDockerContainer,
  getDockerContainerLogs,
} from "@/lib/api";
import type { DockerContainerInfo } from "@/types/api";

interface DockerContainerCardProps {
  container: DockerContainerInfo;
  onActionComplete?: () => void;
}

export default function DockerContainerCard({
  container,
  onActionComplete,
}: DockerContainerCardProps) {
  const [loading, setLoading] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showLogs, setShowLogs] = useState(false);
  const [logs, setLogs] = useState<string>("");
  const [loadingLogs, setLoadingLogs] = useState(false);

  const handleAction = async (action: "start" | "stop" | "restart") => {
    setLoading(action);
    setError(null);

    try {
      await controlDockerContainer(container.name, action);
      onActionComplete?.();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to control container");
    } finally {
      setLoading(null);
      setTimeout(() => setError(null), 3000);
    }
  };

  const handleViewLogs = async () => {
    if (showLogs) {
      setShowLogs(false);
      return;
    }

    setLoadingLogs(true);
    try {
      const response = await getDockerContainerLogs(container.name, 100);
      setLogs(response.logs);
      setShowLogs(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to fetch logs");
    } finally {
      setLoadingLogs(false);
    }
  };

  const isRunning = container.status.toLowerCase() === "running";

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6 border border-gray-200 dark:border-gray-700">
      <div className="flex items-start justify-between mb-4">
        <div className="flex-1">
          <div className="flex items-center gap-3 mb-2">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
              {container.name}
            </h3>
            <StatusBadge status={container.status} />
          </div>
          <p className="text-sm text-gray-500 dark:text-gray-400 font-mono mb-1">
            {container.id.substring(0, 12)}
          </p>
          <p className="text-sm text-gray-600 dark:text-gray-300">
            {container.image}
          </p>
          <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
            Created: {new Date(container.created).toLocaleString()}
          </p>
        </div>
      </div>

      <div className="flex items-center gap-2 flex-wrap">
        {isRunning ? (
          <>
            <button
              onClick={() => handleAction("stop")}
              disabled={loading !== null}
              className="px-3 py-1.5 bg-red-600 text-white rounded hover:bg-red-700 disabled:opacity-50 disabled:cursor-not-allowed text-sm font-medium"
            >
              {loading === "stop" ? "Stopping..." : "Stop"}
            </button>
            <button
              onClick={() => handleAction("restart")}
              disabled={loading !== null}
              className="px-3 py-1.5 bg-yellow-600 text-white rounded hover:bg-yellow-700 disabled:opacity-50 disabled:cursor-not-allowed text-sm font-medium"
            >
              {loading === "restart" ? "Restarting..." : "Restart"}
            </button>
          </>
        ) : (
          <button
            onClick={() => handleAction("start")}
            disabled={loading !== null}
            className="px-3 py-1.5 bg-green-600 text-white rounded hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed text-sm font-medium"
          >
            {loading === "start" ? "Starting..." : "Start"}
          </button>
        )}
        <button
          onClick={handleViewLogs}
          disabled={loadingLogs}
          className="px-3 py-1.5 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed text-sm font-medium"
        >
          {loadingLogs ? "Loading..." : showLogs ? "Hide Logs" : "View Logs"}
        </button>
      </div>

      {error && (
        <div className="mt-2 text-red-600 text-sm bg-red-50 dark:bg-red-900/20 px-2 py-1 rounded">
          {error}
        </div>
      )}

      {showLogs && (
        <div className="mt-4">
          <pre className="bg-gray-900 text-green-400 p-4 rounded-lg overflow-auto font-mono text-sm max-h-[300px]">
            {logs || "No logs available"}
          </pre>
        </div>
      )}
    </div>
  );
}
