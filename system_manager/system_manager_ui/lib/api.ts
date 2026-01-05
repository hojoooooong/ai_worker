import axios, { AxiosError } from "axios";
import type {
  ContainerListResponse,
  ServiceListResponse,
  ServiceStatusResponse,
  ServiceControlResponse,
  ServiceLogsResponse,
  ServiceActionRequest,
  DockerContainerListResponse,
  DockerContainerStatus,
  DockerContainerActionRequest,
  DockerContainerActionResponse,
  DockerContainerLogsResponse,
  ErrorResponse,
} from "@/types/api";

// Get API base URL from environment variable, default to system_manager service name
const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8080";

// Create axios instance with default config
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    "Content-Type": "application/json",
  },
});

// Error handler
function handleError(error: unknown): never {
  if (axios.isAxiosError(error)) {
    const axiosError = error as AxiosError<ErrorResponse>;
    const message =
      axiosError.response?.data?.error ||
      axiosError.message ||
      "An unknown error occurred";
    throw new Error(message);
  }
  throw error;
}

// Container Management

export async function getContainers(): Promise<ContainerListResponse> {
  try {
    const response = await apiClient.get<ContainerListResponse>("/containers");
    return response.data;
  } catch (error) {
    handleError(error);
  }
}

export async function getServices(
  container: string
): Promise<ServiceListResponse> {
  try {
    const response = await apiClient.get<ServiceListResponse>(
      `/containers/${container}/services`
    );
    return response.data;
  } catch (error) {
    handleError(error);
  }
}

export async function getServiceStatus(
  container: string,
  service: string
): Promise<ServiceStatusResponse> {
  try {
    const response = await apiClient.get<ServiceStatusResponse>(
      `/containers/${container}/services/${service}/status`
    );
    return response.data;
  } catch (error) {
    handleError(error);
  }
}

export async function controlService(
  container: string,
  service: string,
  action: "up" | "down" | "restart"
): Promise<ServiceControlResponse> {
  try {
    const request: ServiceActionRequest = { action };
    const response = await apiClient.post<ServiceControlResponse>(
      `/containers/${container}/services/${service}`,
      request
    );
    return response.data;
  } catch (error) {
    handleError(error);
  }
}

export async function getServiceLogs(
  container: string,
  service: string,
  tail: number = 100
): Promise<ServiceLogsResponse> {
  try {
    const response = await apiClient.get<ServiceLogsResponse>(
      `/containers/${container}/services/${service}/logs`,
      {
        params: { tail },
      }
    );
    return response.data;
  } catch (error) {
    handleError(error);
  }
}

// Docker Container Management

export async function getDockerContainers(
  all: boolean = false
): Promise<DockerContainerListResponse> {
  try {
    const response = await apiClient.get<DockerContainerListResponse>(
      "/docker/containers",
      {
        params: { all },
      }
    );
    return response.data;
  } catch (error) {
    handleError(error);
  }
}

export async function getDockerContainerStatus(
  name: string
): Promise<DockerContainerStatus> {
  try {
    const response = await apiClient.get<DockerContainerStatus>(
      `/docker/containers/${name}/status`
    );
    return response.data;
  } catch (error) {
    handleError(error);
  }
}

export async function controlDockerContainer(
  name: string,
  action: "start" | "stop" | "restart",
  timeout?: number
): Promise<DockerContainerActionResponse> {
  try {
    const request: DockerContainerActionRequest = { action, timeout };
    const response = await apiClient.post<DockerContainerActionResponse>(
      `/docker/containers/${name}`,
      request
    );
    return response.data;
  } catch (error) {
    handleError(error);
  }
}

export async function getDockerContainerLogs(
  name: string,
  tail: number = 100
): Promise<DockerContainerLogsResponse> {
  try {
    const response = await apiClient.get<DockerContainerLogsResponse>(
      `/docker/containers/${name}/logs`,
      {
        params: { tail },
      }
    );
    return response.data;
  } catch (error) {
    handleError(error);
  }
}
