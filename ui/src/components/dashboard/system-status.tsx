
import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Cpu, HardDrive, Network, Activity, CheckCircle2, AlertTriangle, AlertOctagon, RefreshCw } from "lucide-react";
import { format } from "date-fns";

export default function SystemStatus() {
  const [cpuUsage, setCpuUsage] = useState(0);
  const [memoryUsage, setMemoryUsage] = useState(0);
  const [apiLatency, setApiLatency] = useState(0);
  const [isLoading, setIsLoading] = useState(true);
  const [systemStatus, setSystemStatus] = useState<"Online" | "Degraded" | "Offline">("Online");
  const [lastUpdated, setLastUpdated] = useState<Date>(new Date());
  const [isRefreshing, setIsRefreshing] = useState(false);

  const simulateMetrics = () => {
    setIsLoading(true);
    setIsRefreshing(true);
    
    // Simulate delay in fetching
    setTimeout(() => {
      // Randomize CPU usage between 15-85%
      const newCpuUsage = Math.floor(Math.random() * 40) + 15;
      // Randomize memory usage between 20-75%
      const newMemoryUsage = Math.floor(Math.random() * 35) + 20;
      // Randomize API latency between 20-250ms
      const newApiLatency = Math.floor(Math.random() * 230) + 20;
      
      // Determine system status based on metrics
      let newStatus: "Online" | "Degraded" | "Offline" = "Online";
      if (newCpuUsage > 80 || newMemoryUsage > 85 || newApiLatency > 300) {
        newStatus = "Degraded";
      }
      
      setCpuUsage(newCpuUsage);
      setMemoryUsage(newMemoryUsage);
      setApiLatency(newApiLatency);
      setSystemStatus(newStatus);
      setLastUpdated(new Date());
      setIsLoading(false);
      setIsRefreshing(false);
    }, 1200);
  };

  // Simulated system monitoring
  useEffect(() => {
    simulateMetrics();
    
    // Update metrics every 10 seconds
    const interval = setInterval(simulateMetrics, 10000);
    return () => clearInterval(interval);
  }, []);

  // Helper to get color classes based on value and thresholds
  const getProgressColorClass = (value: number, warningThreshold: number, criticalThreshold: number) => {
    if (value >= criticalThreshold) return "bg-red-500";
    if (value >= warningThreshold) return "bg-amber-500";
    return "bg-green-500";
  };

  const getStatusIcon = () => {
    if (systemStatus === "Online") return <CheckCircle2 className="h-3 w-3 mr-1" />;
    if (systemStatus === "Degraded") return <AlertTriangle className="h-3 w-3 mr-1" />;
    return <AlertOctagon className="h-3 w-3 mr-1" />;
  };

  // Manual refresh handler
  const handleRefresh = () => {
    simulateMetrics();
  };

  return (
    <Card className="h-full">
      <CardHeader className="pb-2">
        <div className="flex justify-between items-center">
          <CardTitle className="text-lg font-medium">System Status</CardTitle>
          <Badge 
            variant="outline" 
            className={`
              ${systemStatus === "Online" 
                ? "bg-green-500/10 text-green-500 border-green-500/20" 
                : systemStatus === "Degraded" 
                ? "bg-amber-500/10 text-amber-500 border-amber-500/20" 
                : "bg-red-500/10 text-red-500 border-red-500/20"
              }
            `}
          >
            {getStatusIcon()}
            {systemStatus}
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {isLoading ? (
          <div className="space-y-6">
            <div className="space-y-2">
              <div className="h-4 bg-muted animate-pulse rounded w-1/4"></div>
              <div className="h-2 bg-muted animate-pulse rounded"></div>
            </div>
            
            <div className="space-y-2">
              <div className="h-4 bg-muted animate-pulse rounded w-1/3"></div>
              <div className="h-2 bg-muted animate-pulse rounded"></div>
            </div>
            
            <div className="space-y-2">
              <div className="h-4 bg-muted animate-pulse rounded w-1/4"></div>
              <div className="h-2 bg-muted animate-pulse rounded"></div>
            </div>
          </div>
        ) : (
          <>
            <div className="space-y-2">
              <div className="flex justify-between items-center">
                <div className="flex items-center gap-2">
                  <Cpu className="h-4 w-4 text-muted-foreground" />
                  <span className="text-sm font-medium">CPU Usage</span>
                </div>
                <div className="flex items-center">
                  {cpuUsage > 60 && cpuUsage <= 80 && <AlertTriangle className="h-3 w-3 text-amber-500 mr-1" />}
                  {cpuUsage > 80 && <AlertOctagon className="h-3 w-3 text-red-500 mr-1" />}
                  <span className={`text-xs ${
                    cpuUsage > 80 ? "text-red-500" : cpuUsage > 60 ? "text-amber-500" : "text-muted-foreground"
                  }`}>
                    {cpuUsage}%
                  </span>
                </div>
              </div>
              <Progress 
                value={cpuUsage} 
                className="h-2"
                indicatorClassName={getProgressColorClass(cpuUsage, 60, 80)}
              />
            </div>
            
            <div className="space-y-2">
              <div className="flex justify-between items-center">
                <div className="flex items-center gap-2">
                  <HardDrive className="h-4 w-4 text-muted-foreground" />
                  <span className="text-sm font-medium">Memory Usage</span>
                </div>
                <div className="flex items-center">
                  {memoryUsage > 60 && memoryUsage <= 80 && <AlertTriangle className="h-3 w-3 text-amber-500 mr-1" />}
                  {memoryUsage > 80 && <AlertOctagon className="h-3 w-3 text-red-500 mr-1" />}
                  <span className={`text-xs ${
                    memoryUsage > 80 ? "text-red-500" : memoryUsage > 60 ? "text-amber-500" : "text-muted-foreground"
                  }`}>
                    {memoryUsage}%
                  </span>
                </div>
              </div>
              <Progress 
                value={memoryUsage} 
                className="h-2"
                indicatorClassName={getProgressColorClass(memoryUsage, 60, 80)}
              />
            </div>
            
            <div className="space-y-2">
              <div className="flex justify-between items-center">
                <div className="flex items-center gap-2">
                  <Network className="h-4 w-4 text-muted-foreground" />
                  <span className="text-sm font-medium">API Latency</span>
                </div>
                <div className="flex items-center">
                  {apiLatency > 200 && apiLatency <= 300 && <AlertTriangle className="h-3 w-3 text-amber-500 mr-1" />}
                  {apiLatency > 300 && <AlertOctagon className="h-3 w-3 text-red-500 mr-1" />}
                  <span className={`text-xs ${
                    apiLatency > 300 ? "text-red-500" : apiLatency > 200 ? "text-amber-500" : "text-muted-foreground"
                  }`}>
                    {apiLatency}ms
                  </span>
                </div>
              </div>
              <Progress 
                value={apiLatency / 5} // Scale to fit in progress bar (max would be ~200-300ms)
                className="h-2"
                indicatorClassName={getProgressColorClass(apiLatency, 200, 300)}
              />
            </div>
            
            <div className="pt-2">
              <div className="flex items-center gap-2">
                <Activity className="h-4 w-4 text-muted-foreground" />
                <span className="text-sm font-medium">Bot Status</span>
              </div>
              <div className="flex justify-between items-center mt-1">
                <div className="text-xs text-muted-foreground">
                  Last updated: {format(lastUpdated, "HH:mm:ss")}
                </div>
                <Button 
                  variant="outline" 
                  size="sm" 
                  className="h-7 px-2" 
                  onClick={handleRefresh}
                  disabled={isRefreshing}
                >
                  <RefreshCw className={`h-3 w-3 mr-1 ${isRefreshing ? 'animate-spin' : ''}`} />
                  Refresh
                </Button>
              </div>
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
}
