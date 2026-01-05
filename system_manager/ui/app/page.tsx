"use client";

import { useEffect, useState } from "react";
import ContainerCard from "@/components/ContainerCard";
import { getContainers } from "@/lib/api";
import type { ContainerInfo } from "@/types/api";

export default function HomePage() {
  const [containers, setContainers] = useState<ContainerInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadContainers();
  }, []);

  const loadContainers = async () => {
    try {
      setLoading(true);
      setError(null);
      const response = await getContainers();
      setContainers(response.containers);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load containers");
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="flex justify-center items-center h-64">
        <div className="text-gray-600 dark:text-gray-400">Loading containers...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-red-800 dark:text-red-200 font-medium">
              Error loading containers
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
      <div className="mb-6">
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
          Containers
        </h1>
        <p className="text-gray-600 dark:text-gray-400 mt-2">
          Manage ROS2-based robot containers and their services
        </p>
      </div>

      {containers.length === 0 ? (
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-8 text-center border border-gray-200 dark:border-gray-700">
          <p className="text-gray-600 dark:text-gray-400">
            No containers configured
          </p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {containers.map((container) => (
            <ContainerCard key={container.name} container={container} />
          ))}
        </div>
      )}
    </div>
  );
}
