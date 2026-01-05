"use client";

import Link from "next/link";
import type { ContainerInfo } from "@/types/api";

interface ContainerCardProps {
  container: ContainerInfo;
}

export default function ContainerCard({ container }: ContainerCardProps) {
  return (
    <Link href={`/containers/${container.name}`}>
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md hover:shadow-lg transition-shadow p-6 border border-gray-200 dark:border-gray-700">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
              {container.name}
            </h3>
            <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
              {container.socket_path}
            </p>
          </div>
          <svg
            className="w-5 h-5 text-gray-400"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M9 5l7 7-7 7"
            />
          </svg>
        </div>
      </div>
    </Link>
  );
}
