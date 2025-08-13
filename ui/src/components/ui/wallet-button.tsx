
import { useState, useEffect, useMemo } from "react";
import { Button } from "@/components/ui/button";
import { Wallet, Copy, ChevronDown, ExternalLink, AlertTriangle } from "lucide-react";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { toast } from "sonner";

// Solana imports
import { WalletReadyState } from "@solana/wallet-adapter-base";
import { useWallet } from "@solana/wallet-adapter-react";
import { useWalletModal } from "@solana/wallet-adapter-react-ui";

interface WalletButtonProps {
  className?: string;
}

export function WalletButton({ className }: WalletButtonProps) {
  const { publicKey, wallet, disconnect, connecting, connected } = useWallet();
  const { setVisible } = useWalletModal();
  const [copied, setCopied] = useState(false);
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);

  const walletAddress = publicKey?.toBase58() || "";
  const shortAddress = walletAddress ? `${walletAddress.slice(0, 4)}...${walletAddress.slice(-4)}` : "";
  
  const isWalletReady = useMemo(() => 
    wallet?.readyState === WalletReadyState.Installed || 
    wallet?.readyState === WalletReadyState.Loadable, 
    [wallet]
  );
  
  const handleConnect = async () => {
    if (connected) return;
    setVisible(true);
  };
  
  const handleDisconnect = () => {
    if (connected) {
      disconnect();
      toast.info("Wallet disconnected");
    }
  };
  
  const handleCopy = () => {
    if (!walletAddress) return;
    
    navigator.clipboard.writeText(walletAddress);
    setCopied(true);
    toast.success("Address copied to clipboard");
    
    setTimeout(() => setCopied(false), 2000);
  };
  
  const handleViewExplorer = () => {
    if (!walletAddress) return;
    
    const explorerUrl = `https://explorer.solana.com/address/${walletAddress}`;
    window.open(explorerUrl, "_blank");
    toast.info("Opening explorer in new tab");
  };
  
  // Auto-close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = () => {
      if (isDropdownOpen) {
        setIsDropdownOpen(false);
      }
    };
    
    document.addEventListener("click", handleClickOutside);
    return () => {
      document.removeEventListener("click", handleClickOutside);
    };
  }, [isDropdownOpen]);

  return (
    <div className={className}>
      {!connected ? (
        <Button
          onClick={handleConnect}
          disabled={connecting}
          variant="outline"
          className="bg-solana/10 hover:bg-solana/20 border-solana/20 text-solana font-medium"
        >
          <Wallet className="mr-2 h-4 w-4" />
          {connecting ? "Connecting..." : "Connect Wallet"}
        </Button>
      ) : (
        <DropdownMenu open={isDropdownOpen} onOpenChange={setIsDropdownOpen}>
          <DropdownMenuTrigger asChild onClick={(e) => e.stopPropagation()}>
            <Button 
              variant="outline" 
              className="bg-solana/10 hover:bg-solana/20 border-solana/20 text-solana font-medium"
            >
              <Wallet className="mr-2 h-4 w-4" />
              {shortAddress}
              <ChevronDown className="ml-2 h-4 w-4" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end" className="min-w-[240px]">
            <DropdownMenuLabel className="flex items-center">
              <span className="flex-1">Connected Wallet</span>
              {wallet && (
                <span className="text-xs text-muted-foreground">{wallet.adapter.name}</span>
              )}
            </DropdownMenuLabel>
            <DropdownMenuSeparator />
            <DropdownMenuItem onClick={handleCopy} className="cursor-pointer flex justify-between">
              <span className="flex items-center">
                <Copy className="mr-2 h-4 w-4" /> Copy Address
              </span>
              {copied && <span className="text-xs text-green-500">Copied!</span>}
            </DropdownMenuItem>
            <DropdownMenuItem onClick={handleViewExplorer} className="cursor-pointer">
              <ExternalLink className="mr-2 h-4 w-4" /> View on Explorer
            </DropdownMenuItem>
            <DropdownMenuSeparator />
            <DropdownMenuItem 
              onClick={handleDisconnect} 
              className="cursor-pointer text-red-500 focus:text-red-500"
            >
              Disconnect
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      )}
    </div>
  );
}
