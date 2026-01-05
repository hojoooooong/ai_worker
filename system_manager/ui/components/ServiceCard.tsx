"use client";

import Link from "next/link";
import StatusBadge from "./StatusBadge";
import ServiceControls from "./ServiceControls";
import type { ServiceInfo, ServiceStatusResponse } from "@/types/api";

interface ServiceCardProps {
  container: string;
  service: ServiceInfo;
  status?: ServiceStatusResponse;
  onStatusUpdate?: () => void;
}

export default function ServiceCard({
  container,
  service,
  status,
  onStatusUpdate,
}: ServiceCardProps) {
  const isUp = status?.is_up ?? false;
  const uptime = status?.uptime_seconds;

  const formatUptime = (seconds: number | null | undefined): string => {
    if (!seconds) return "";
    if (seconds < 60) return `${seconds}s`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m`;
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    return `${hours}h ${minutes}m`;
  };

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6 border border-gray-200 dark:border-gray-700">
      <div className="flex items-start justify-between mb-4">
        <div className="flex-1">
          <div className="flex items-center gap-3 mb-2">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
              {service.label}
            </h3>
            <StatusBadge status={isUp} />
          </div>
          <p className="text-sm text-gray-500 dark:text-gray-400 font-mono">
            {service.id}
          </p>
          {status && (
            <div className="mt-2 text-sm text-gray-600 dark:text-gray-300">
              {status.pid && (
                <span className="mr-4">PID: {status.pid}</span>
              )}
              {uptime !== null && uptime !== undefined && (
                <span>Uptime: {formatUptime(uptime)}</span>
              )}
            </div>
          )}
        </div>
      </div>

      <div className="flex items-center justify-between gap-4">
        <ServiceControls
          container={container}
          service={service.id}
          onActionComplete={onStatusUpdate}
        />
        <Link
          href={`/containers/${container}/services/${service.id}/logs`}
          className="px-3 py-1.5 bg-blue-600 text-white rounded hover:bg-blue-700 text-sm font-medium"
        >
          View Logs
        </Link>
      </div>
    </div>
  );
}
