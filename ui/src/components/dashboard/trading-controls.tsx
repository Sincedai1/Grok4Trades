
import { useState } from "react";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { Slider } from "@/components/ui/slider";
import { Badge } from "@/components/ui/badge";
import { AlertCircle, Play, Pause, StopCircle, RefreshCw } from "lucide-react";
import { toast } from "sonner";
import { useToast } from "@/hooks/use-toast";

export default function TradingControls() {
  const [isActive, setIsActive] = useState(false);
  const [riskLevel, setRiskLevel] = useState(40);
  const [allocatedBalance, setAllocatedBalance] = useState(60);
  const [isPending, setIsPending] = useState(false);
  const [isEmergencyStop, setIsEmergencyStop] = useState(false);
  const { toast: shadcnToast } = useToast();

  const handleStart = () => {
    if (isEmergencyStop) {
      toast.error("Trading bot is in emergency stop mode. Please reset to continue.", {
        description: "Check logs for more information on what triggered the stop.",
        duration: 5000,
      });
      return;
    }
    
    setIsPending(true);
    // Simulate delay
    setTimeout(() => {
      setIsActive(true);
      setIsPending(false);
      toast.success("Trading bot activated", {
        description: "Using AI-powered strategies to analyze market conditions",
      });
    }, 2000);
  };

  const handleStop = () => {
    setIsPending(true);
    // Simulate delay
    setTimeout(() => {
      setIsActive(false);
      setIsPending(false);
      toast.info("Trading bot deactivated", {
        description: "All active positions remain open",
      });
    }, 1500);
  };

  const handleEmergencyStop = () => {
    setIsPending(true);
    shadcnToast({
      variant: "destructive",
      title: "Emergency Stop Triggered",
      description: "Liquidating all positions and halting all trading activity"
    });
    
    // Simulate delay
    setTimeout(() => {
      setIsActive(false);
      setIsEmergencyStop(true);
      setIsPending(false);
      toast.error("EMERGENCY STOP TRIGGERED", {
        description: "All positions liquidated and trading halted",
        duration: 6000,
      });
    }, 2000);
  };

  const handleReset = () => {
    setIsEmergencyStop(false);
    toast.success("Trading bot reset", {
      description: "Ready to be activated",
    });
  };

  return (
    <Card className="h-full">
      <CardHeader>
        <CardTitle className="text-lg font-medium flex items-center">
          Bot Controls
          <Badge
            variant="outline"
            className={`ml-2 ${
              isActive
                ? "bg-green-500/10 text-green-500 border-green-500/20"
                : isEmergencyStop
                ? "bg-red-500/10 text-red-500 border-red-500/20"
                : "bg-gray-500/10 text-gray-500 border-gray-500/20"
            }`}
          >
            {isActive ? "Running" : isEmergencyStop ? "Emergency Stop" : "Idle"}
          </Badge>
        </CardTitle>
        <CardDescription>Configure and control your Solana trading bot</CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        {isEmergencyStop && (
          <div className="bg-red-500/10 border border-red-500/20 rounded-md p-3 flex items-start space-x-3">
            <AlertCircle className="text-red-500 h-5 w-5 mt-0.5" />
            <div>
              <h4 className="text-sm font-medium text-red-500">Emergency Stop Active</h4>
              <p className="text-xs text-red-400 mt-1">
                The bot has been emergency stopped due to unusual market activity or risk detection.
                Reset the bot to continue trading.
              </p>
            </div>
          </div>
        )}

        <div className="space-y-4">
          <div>
            <div className="flex justify-between mb-2">
              <span className="text-sm font-medium">Risk Level</span>
              <span className="text-sm text-muted-foreground">{riskLevel}%</span>
            </div>
            <Slider
              value={[riskLevel]}
              min={10}
              max={90}
              step={5}
              onValueChange={(value) => setRiskLevel(value[0])}
              disabled={isActive || isPending}
              className={isActive ? "opacity-70" : ""}
            />
            <div className="flex justify-between text-xs text-muted-foreground mt-1">
              <span>Conservative</span>
              <span>Aggressive</span>
            </div>
          </div>

          <div>
            <div className="flex justify-between mb-2">
              <span className="text-sm font-medium">Allocated Balance</span>
              <span className="text-sm text-muted-foreground">{allocatedBalance}%</span>
            </div>
            <Slider
              value={[allocatedBalance]}
              min={10}
              max={100}
              step={5}
              onValueChange={(value) => setAllocatedBalance(value[0])}
              disabled={isActive || isPending}
              className={isActive ? "opacity-70" : ""}
            />
            <div className="flex justify-between text-xs text-muted-foreground mt-1">
              <span>10%</span>
              <span>100%</span>
            </div>
          </div>

          <div className="flex items-center justify-between">
            <div className="space-y-0.5">
              <div className="text-sm font-medium">AI-Enhanced Trading</div>
              <div className="text-xs text-muted-foreground">
                Use GPT-4o for trading decisions
              </div>
            </div>
            <Switch
              checked={true}
              disabled
            />
          </div>
        </div>
      </CardContent>
      <CardFooter className="flex flex-col space-y-2">
        <div className="grid grid-cols-2 gap-2 w-full">
          {!isActive ? (
            <Button
              className="w-full bg-solana hover:bg-solana/90"
              onClick={handleStart}
              disabled={isPending || isEmergencyStop}
            >
              {isPending ? (
                <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
              ) : (
                <Play className="h-4 w-4 mr-2" />
              )}
              Start Bot
            </Button>
          ) : (
            <Button
              className="w-full bg-amber-500 hover:bg-amber-600"
              onClick={handleStop}
              disabled={isPending || isEmergencyStop}
            >
              {isPending ? (
                <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
              ) : (
                <Pause className="h-4 w-4 mr-2" />
              )}
              Pause Bot
            </Button>
          )}
          
          {!isEmergencyStop ? (
            <Button
              variant="destructive"
              className="w-full"
              onClick={handleEmergencyStop}
              disabled={isPending}
            >
              <StopCircle className="h-4 w-4 mr-2" />
              Emergency Stop
            </Button>
          ) : (
            <Button
              variant="outline"
              className="w-full"
              onClick={handleReset}
            >
              <RefreshCw className="h-4 w-4 mr-2" />
              Reset Bot
            </Button>
          )}
        </div>
      </CardFooter>
    </Card>
  );
}
