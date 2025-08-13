import { useEffect, useState } from 'react';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export interface Limits {
  min_notional: number;
  max_notional_per_order: number;
  max_daily_notional: number;
  dry_run: boolean;
  alerts_enabled: boolean;
  kill_switch_active: boolean;
}

export interface Order {
  order_id: string;
  client_order_id: string;
  symbol: string;
  side: string;
  amount: number;
  price?: number;
  status: string;
  timestamp: string;
  exchange: string;
}

export interface StreamEvent {
  type: string;
  data?: any;
  timestamp: string;
}

export function useApiClient() {
  const [limits, setLimits] = useState<Limits | null>(null);
  const [orders, setOrders] = useState<Order[]>([]);
  const [events, setEvents] = useState<StreamEvent[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Fetch limits
  const fetchLimits = async () => {
    try {
      const response = await fetch(`${API_URL}/api/limits`);
      if (!response.ok) throw new Error('Failed to fetch limits');
      const data = await response.json();
      setLimits(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    }
  };

  // Fetch orders
  const fetchOrders = async () => {
    try {
      const response = await fetch(`${API_URL}/api/orders`);
      if (!response.ok) throw new Error('Failed to fetch orders');
      const data = await response.json();
      setOrders(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    }
  };

  // Create order
  const createOrder = async (order: {
    symbol: string;
    side: string;
    amount: number;
    order_type?: string;
    price?: number;
  }) => {
    try {
      const response = await fetch(`${API_URL}/api/orders`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(order),
      });
      if (!response.ok) throw new Error('Failed to create order');
      const data = await response.json();
      await fetchOrders(); // Refresh orders
      return data;
    } catch (err) {
      throw err instanceof Error ? err : new Error('Unknown error');
    }
  };

  // Setup SSE connection
  useEffect(() => {
    const eventSource = new EventSource(`${API_URL}/stream/events`);

    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        setEvents((prev) => [...prev.slice(-99), data]); // Keep last 100 events
        
        // Update orders if order event
        if (data.type === 'order_status') {
          fetchOrders();
        }
      } catch (err) {
        console.error('Failed to parse event:', err);
      }
    };

    eventSource.onerror = (error) => {
      console.error('SSE error:', error);
      eventSource.close();
      
      // Retry after 5 seconds
      setTimeout(() => {
        window.location.reload();
      }, 5000);
    };

    return () => {
      eventSource.close();
    };
  }, []);

  // Initial load
  useEffect(() => {
    const loadData = async () => {
      setLoading(true);
      await Promise.all([fetchLimits(), fetchOrders()]);
      setLoading(false);
    };
    
    loadData();
  }, []);

  return {
    limits,
    orders,
    events,
    loading,
    error,
    createOrder,
    refresh: () => {
      fetchLimits();
      fetchOrders();
    },
  };
}
