"use client";

import { useEffect, useRef, useState } from "react";

interface LogViewerProps {
  logs: string;
  autoScroll?: boolean;
  className?: string;
}

export default function LogViewer({
  logs,
  autoScroll = true,
  className = "",
}: LogViewerProps) {
  const logRef = useRef<HTMLPreElement>(null);
  const [isScrolledToBottom, setIsScrolledToBottom] = useState(true);

  useEffect(() => {
    if (autoScroll && isScrolledToBottom && logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [logs, autoScroll, isScrolledToBottom]);

  const handleScroll = () => {
    if (logRef.current) {
      const { scrollTop, scrollHeight, clientHeight } = logRef.current;
      const isAtBottom = scrollTop + clientHeight >= scrollHeight - 10;
      setIsScrolledToBottom(isAtBottom);
    }
  };

  return (
    <div className={`relative ${className}`}>
      <pre
        ref={logRef}
        onScroll={handleScroll}
        className="bg-gray-900 text-green-400 p-4 rounded-lg overflow-auto font-mono text-sm h-full max-h-[600px]"
        style={{ fontFamily: "monospace" }}
      >
        {logs || "No logs available"}
      </pre>
      {!isScrolledToBottom && autoScroll && (
        <button
          onClick={() => {
            if (logRef.current) {
              logRef.current.scrollTop = logRef.current.scrollHeight;
              setIsScrolledToBottom(true);
            }
          }}
          className="absolute bottom-4 right-4 bg-blue-600 text-white px-3 py-1 rounded text-sm hover:bg-blue-700"
        >
          Scroll to bottom
        </button>
      )}
    </div>
  );
}
