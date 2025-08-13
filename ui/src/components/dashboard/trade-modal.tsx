
import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { AlertCircle, Check, RefreshCw } from "lucide-react";
import { toast } from "sonner";
import { useWallet } from "@solana/wallet-adapter-react";
import { executeTradeOnChain, getAIRecommendedEntry, getCurrentPrice, getAIRiskAssessment } from "@/lib/trade-execution";

interface TradeModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  defaultToken?: string;
  defaultSide?: 'buy' | 'sell';
}

export function TradeModal({ 
  open, 
  onOpenChange,
  defaultToken = "SOL",
  defaultSide = "buy"
}: TradeModalProps) {
  const { publicKey, connected } = useWallet();
  const [token, setToken] = useState(defaultToken);
  const [side, setSide] = useState<'buy' | 'sell'>(defaultSide);
  const [amount, setAmount] = useState(1);
  const [slippage, setSlippage] = useState(1);
  const [isLoadingPrice, setIsLoadingPrice] = useState(false);
  const [currentPrice, setCurrentPrice] = useState(0);
  const [aiRecommendedPrice, setAiRecommendedPrice] = useState(0);
  const [riskScore, setRiskScore] = useState(0);
  const [isExecuting, setIsExecuting] = useState(false);
  
  // Available tokens for trading
  const availableTokens = [
    { symbol: "SOL", name: "Solana" },
    { symbol: "JUP", name: "Jupiter" },
    { symbol: "BONK", name: "Bonk" },
    { symbol: "JTO", name: "Jito" },
    { symbol: "PYTH", name: "Pyth Network" },
    { symbol: "WIF", name: "Dogwifhat" },
  ];

  useEffect(() => {
    if (open && token) {
      loadTokenData(token);
    }
  }, [open, token]);

  const loadTokenData = async (selectedToken: string) => {
    setIsLoadingPrice(true);
    try {
      const [price, aiPrice] = await Promise.all([
        getCurrentPrice(selectedToken),
        getAIRecommendedEntry(selectedToken)
      ]);
      
      setCurrentPrice(price);
      setAiRecommendedPrice(aiPrice);
      setRiskScore(getAIRiskAssessment(selectedToken));
    } catch (error) {
      console.error("Error loading token data:", error);
      toast.error("Failed to load market data");
    } finally {
      setIsLoadingPrice(false);
    }
  };

  const handleExecuteTrade = async () => {
    if (!connected || !publicKey) {
      toast.error("Please connect your wallet first");
      return;
    }

    setIsExecuting(true);
    try {
      const result = await executeTradeOnChain({
        tokenSymbol: token,
        amount: amount,
        price: currentPrice,
        side: side,
        walletAddress: publicKey.toString()
      });

      if (result.success) {
        toast.success(`${side.toUpperCase()} order executed successfully`, {
          description: `${amount} ${token} at $${currentPrice.toFixed(4)}`,
        });
        onOpenChange(false);
      }
    } catch (error) {
      console.error("Trade execution error:", error);
      toast.error("Trade execution failed");
    } finally {
      setIsExecuting(false);
    }
  };

  const getRiskLabel = (score: number) => {
    if (score < 30) return { label: "Low Risk", color: "bg-green-500" };
    if (score < 70) return { label: "Medium Risk", color: "bg-yellow-500" };
    return { label: "High Risk", color: "bg-red-500" };
  };

  const risk = getRiskLabel(riskScore);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[425px]">
        <DialogHeader>
          <DialogTitle>Execute AI-Assisted Trade</DialogTitle>
          <DialogDescription>
            Use our AI to optimize your trade parameters and execution.
          </DialogDescription>
        </DialogHeader>
        
        {!connected && (
          <Alert variant="destructive" className="mt-2">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>
              Please connect your wallet first to execute trades.
            </AlertDescription>
          </Alert>
        )}

        <div className="grid gap-4 py-4">
          <div className="grid grid-cols-4 items-center gap-4">
            <Label htmlFor="token" className="text-right">
              Token
            </Label>
            <Select
              value={token}
              onValueChange={(value) => {
                setToken(value);
                loadTokenData(value);
              }}
            >
              <SelectTrigger id="token" className="col-span-3">
                <SelectValue placeholder="Select token" />
              </SelectTrigger>
              <SelectContent>
                <SelectGroup>
                  <SelectLabel>Popular Tokens</SelectLabel>
                  {availableTokens.map((token) => (
                    <SelectItem key={token.symbol} value={token.symbol}>
                      {token.symbol} - {token.name}
                    </SelectItem>
                  ))}
                </SelectGroup>
              </SelectContent>
            </Select>
          </div>

          <div className="grid grid-cols-4 items-center gap-4">
            <Label htmlFor="side" className="text-right">
              Side
            </Label>
            <Select
              value={side}
              onValueChange={(value) => setSide(value as 'buy' | 'sell')}
            >
              <SelectTrigger id="side" className="col-span-3">
                <SelectValue placeholder="Select side" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="buy" className="text-green-500">Buy</SelectItem>
                <SelectItem value="sell" className="text-red-500">Sell</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="grid grid-cols-4 items-center gap-4">
            <Label htmlFor="amount" className="text-right">
              Amount
            </Label>
            <Input
              id="amount"
              type="number"
              value={amount}
              onChange={(e) => setAmount(parseFloat(e.target.value) || 0)}
              className="col-span-3"
            />
          </div>

          <div className="grid grid-cols-4 items-center gap-4">
            <Label htmlFor="price" className="text-right">
              Price
            </Label>
            <div className="col-span-3 flex items-center space-x-2">
              <span className="text-lg font-medium">
                ${isLoadingPrice ? "..." : currentPrice.toFixed(4)}
              </span>
              {isLoadingPrice && <RefreshCw className="h-4 w-4 animate-spin" />}
            </div>
          </div>

          <div className="grid grid-cols-4 items-center gap-4">
            <Label htmlFor="slippage" className="text-right">
              Slippage
            </Label>
            <div className="col-span-3">
              <div className="flex justify-between mb-2">
                <span className="text-sm">{slippage}%</span>
              </div>
              <Slider
                value={[slippage]}
                min={0.1}
                max={5}
                step={0.1}
                onValueChange={(value) => setSlippage(value[0])}
              />
            </div>
          </div>

          <div className="col-span-4 mt-2">
            <div className="rounded-md border p-3 bg-muted/30">
              <h4 className="font-medium mb-2 flex items-center">
                <Check className="h-4 w-4 mr-1 text-green-500" />
                AI Trading Insights
              </h4>
              <div className="grid grid-cols-2 gap-2 text-sm">
                <div>
                  <span className="text-muted-foreground">Recommended Price:</span>
                  <div className="font-medium">${aiRecommendedPrice.toFixed(4)}</div>
                </div>
                <div>
                  <span className="text-muted-foreground">Risk Assessment:</span>
                  <div>
                    <Badge className={`${risk.color} text-white`}>
                      {risk.label}
                    </Badge>
                  </div>
                </div>
                <div className="col-span-2 mt-1">
                  <span className="text-muted-foreground">AI Assessment:</span>
                  <div className="font-medium">
                    {riskScore < 30 ? (
                      "Good entry point with favorable risk/reward ratio"
                    ) : riskScore < 70 ? (
                      "Moderate volatility expected, consider smaller position size"
                    ) : (
                      "High risk trade, consider waiting for better entry"
                    )}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
        
        <DialogFooter>
          <Button onClick={() => onOpenChange(false)} variant="outline">
            Cancel
          </Button>
          <Button
            variant={side === "buy" ? "default" : "destructive"}
            onClick={handleExecuteTrade}
            disabled={isExecuting || !connected || amount <= 0}
          >
            {isExecuting ? (
              <>
                <RefreshCw className="mr-2 h-4 w-4 animate-spin" />
                Executing...
              </>
            ) : (
              side === "buy" ? "Buy" : "Sell"
            )}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
