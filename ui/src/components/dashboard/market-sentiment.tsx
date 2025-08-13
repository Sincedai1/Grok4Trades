
import { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

type Sentiment = "Bullish" | "Bearish" | "Neutral" | "Very Bullish" | "Very Bearish";

interface SentimentSource {
  source: string;
  sentiment: Sentiment;
  score: number; // 0-100
}

const sentimentColors = {
  "Very Bullish": "bg-green-500",
  "Bullish": "bg-green-400",
  "Neutral": "bg-gray-400",
  "Bearish": "bg-red-400",
  "Very Bearish": "bg-red-500",
};

export default function MarketSentiment() {
  const [overallSentiment, setOverallSentiment] = useState<Sentiment>("Neutral");
  const [sentimentSources, setSentimentSources] = useState<SentimentSource[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  // Simulate fetching sentiment data
  useEffect(() => {
    const fetchSentiment = () => {
      setIsLoading(true);
      
      // Simulated API response
      setTimeout(() => {
        const mockSources: SentimentSource[] = [
          { source: "Twitter", sentiment: "Bullish", score: 68 },
          { source: "Pump.fun", sentiment: "Very Bullish", score: 82 },
          { source: "Photon-SOL", sentiment: "Neutral", score: 52 },
          { source: "Whale Activity", sentiment: "Bullish", score: 71 },
        ];
        
        // Calculate overall sentiment based on average score
        const avgScore = mockSources.reduce((acc, src) => acc + src.score, 0) / mockSources.length;
        let overall: Sentiment = "Neutral";
        
        if (avgScore > 80) overall = "Very Bullish";
        else if (avgScore > 60) overall = "Bullish";
        else if (avgScore < 20) overall = "Very Bearish";
        else if (avgScore < 40) overall = "Bearish";
        
        setSentimentSources(mockSources);
        setOverallSentiment(overall);
        setIsLoading(false);
      }, 1500);
    };

    fetchSentiment();
    
    // Update sentiment every 2 minutes
    const interval = setInterval(fetchSentiment, 120000);
    return () => clearInterval(interval);
  }, []);

  return (
    <Card className="h-full">
      <CardHeader className="pb-2">
        <CardTitle className="text-lg font-medium">Market Sentiment</CardTitle>
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <div className="space-y-3">
            <div className="h-8 bg-muted animate-pulse rounded"></div>
            <div className="space-y-2">
              {[1, 2, 3, 4].map((i) => (
                <div key={i} className="flex justify-between items-center">
                  <div className="h-4 bg-muted animate-pulse rounded w-1/4"></div>
                  <div className="h-4 bg-muted animate-pulse rounded w-1/5"></div>
                </div>
              ))}
            </div>
          </div>
        ) : (
          <>
            <div className="mb-4 text-center">
              <Badge 
                className={`px-3 py-1 ${sentimentColors[overallSentiment]} text-white`}
              >
                {overallSentiment}
              </Badge>
              <p className="text-sm text-muted-foreground mt-1">
                Overall market sentiment based on AI analysis
              </p>
            </div>
            
            <div className="space-y-2">
              {sentimentSources.map((source) => (
                <div key={source.source} className="flex justify-between items-center">
                  <span className="text-sm font-medium">{source.source}</span>
                  <Badge 
                    variant="outline" 
                    className={`${source.score > 60 ? 'text-green-500' : source.score < 40 ? 'text-red-500' : 'text-gray-500'}`}
                  >
                    {source.sentiment} ({source.score}%)
                  </Badge>
                </div>
              ))}
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
}
