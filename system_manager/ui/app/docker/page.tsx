"use client";

import { useEffect, useState } from "react";
import DockerContainerCard from "@/components/DockerContainerCard";
import { getDockerContainers } from "@/lib/api";
import type { DockerContainerInfo } from "@/types/api";

export default function DockerPage() {
  const [containers, setContainers] = useState<DockerContainerInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showAll, setShowAll] = useState(false);

  useEffect(() => {
    loadContainers();
  }, [showAll]);

  const loadContainers = async () => {
    try {
      setLoading(true);
      setError(null);
      const response = await getDockerContainers(showAll);
      setContainers(response.containers);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load Docker containers");
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="flex justify-center items-center h-64">
        <div className="text-gray-600 dark:text-gray-400">Loading Docker containers...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-red-800 dark:text-red-200 font-medium">
              Error loading Docker containers
            </h3>
            <p className="text-red-600 dark:text-red-300 text-sm mt-1">{error}</p>
          </div>
          <button
            onClick={loadContainers}
            className="px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700 text-sm font-medium"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div>
      <div className="mb-6 flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
            Docker Containers
          </h1>
          <p className="text-gray-600 dark:text-gray-400 mt-2">
            Manage Docker containers on the system
          </p>
        </div>
        <div className="flex items-center gap-2">
          <label className="flex items-center gap-2 text-sm text-gray-700 dark:text-gray-300">
            <input
              type="checkbox"
              checked={showAll}
              onChange={(e) => setShowAll(e.target.checked)}
              className="rounded"
            />
            Show all (including stopped)
          </label>
          <button
            onClick={loadContainers}
            className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 text-sm font-medium"
          >
            Refresh
          </button>
        </div>
      </div>

      {containers.length === 0 ? (
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-8 text-center border border-gray-200 dark:border-gray-700">
          <p className="text-gray-600 dark:text-gray-400">No Docker containers found</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-6">
          {containers.map((container) => (
            <DockerContainerCard
              key={container.id}
              container={container}
              onActionComplete={loadContainers}
            />
          ))}
        </div>
      )}
    </div>
  );
}
