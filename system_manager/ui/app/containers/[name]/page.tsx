"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import ServiceCard from "@/components/ServiceCard";
import { getServices, getServiceStatus } from "@/lib/api";
import type { ServiceInfo, ServiceStatusResponse } from "@/types/api";

export default function ContainerDetailPage() {
  const params = useParams();
  const containerName = params.name as string;

  const [services, setServices] = useState<ServiceInfo[]>([]);
  const [serviceStatuses, setServiceStatuses] = useState<
    Record<string, ServiceStatusResponse>
  >({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);

  useEffect(() => {
    loadData();
  }, [containerName]);

  useEffect(() => {
    if (services.length > 0) {
      // Initial load
      loadStatuses();
      // Set up auto-refresh every 5 seconds
      const interval = setInterval(() => {
        loadStatuses();
      }, 5000);
      return () => clearInterval(interval);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [services.length, containerName]);

  const loadData = async () => {
    try {
      setLoading(true);
      setError(null);
      const servicesResponse = await getServices(containerName);
      setServices(servicesResponse.services);
      await loadStatuses();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load services");
    } finally {
      setLoading(false);
    }
  };

  const loadStatuses = async () => {
    if (services.length === 0) return;
    
    try {
      setRefreshing(true);
      const statusPromises = services.map((service) =>
        getServiceStatus(containerName, service.id).catch(() => null)
      );
      const statuses = await Promise.all(statusPromises);
      const statusMap: Record<string, ServiceStatusResponse> = {};
      statuses.forEach((status, index) => {
        if (status) {
          statusMap[services[index].id] = status;
        }
      });
      setServiceStatuses(statusMap);
    } catch (err) {
      // Silently fail for status updates
    } finally {
      setRefreshing(false);
    }
  };

  const handleStatusUpdate = () => {
    loadStatuses();
  };

  if (loading) {
    return (
      <div className="flex justify-center items-center h-64">
        <div className="text-gray-600 dark:text-gray-400">Loading services...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-red-800 dark:text-red-200 font-medium">
              Error loading services
            </h3>
            <p className="text-red-600 dark:text-red-300 text-sm mt-1">{error}</p>
          </div>
          <button
            onClick={loadData}
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
          <Link
            href="/"
            className="text-blue-600 dark:text-blue-400 hover:underline text-sm mb-2 inline-block"
          >
            ← Back to containers
          </Link>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
            {containerName}
          </h1>
          <p className="text-gray-600 dark:text-gray-400 mt-2">
            Manage services in this container
          </p>
        </div>
        <div className="flex items-center gap-2">
          {refreshing && (
            <span className="text-sm text-gray-500 dark:text-gray-400">
              Refreshing...
            </span>
          )}
          <button
            onClick={loadStatuses}
            className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 text-sm font-medium"
          >
            Refresh
          </button>
        </div>
      </div>

      {services.length === 0 ? (
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-8 text-center border border-gray-200 dark:border-gray-700">
          <p className="text-gray-600 dark:text-gray-400">No services found</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-6">
          {services.map((service) => (
            <ServiceCard
              key={service.id}
              container={containerName}
              service={service}
              status={serviceStatuses[service.id]}
              onStatusUpdate={handleStatusUpdate}
            />
          ))}
        </div>
      )}
    </div>
  );
}
