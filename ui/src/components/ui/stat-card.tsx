
import { cn } from "@/lib/utils";
import { cva, type VariantProps } from "class-variance-authority";
import { HTMLAttributes, ReactNode } from "react";

const statCardVariants = cva(
  "rounded-lg p-4 shadow-sm transition-all ease-in-out duration-200 overflow-hidden",
  {
    variants: {
      variant: {
        default: "bg-card text-card-foreground border",
        glass: "glass border border-white/10",
        gradient: "gradient-animate text-white border-none",
        outline: "border border-border bg-transparent",
      },
      size: {
        default: "p-6",
        sm: "p-4",
        lg: "p-8",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  }
);

export interface StatCardProps
  extends HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof statCardVariants> {
  title: string;
  value: string | number;
  icon?: ReactNode;
  change?: {
    value: string | number;
    positive?: boolean;
  };
  footer?: ReactNode;
}

export function StatCard({
  className,
  variant,
  size,
  title,
  value,
  icon,
  change,
  footer,
  ...props
}: StatCardProps) {
  return (
    <div
      className={cn(statCardVariants({ variant, size, className }))}
      {...props}
    >
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-sm font-medium text-muted-foreground">{title}</h3>
        {icon && <div className="text-muted-foreground">{icon}</div>}
      </div>
      <div className="flex items-baseline gap-2">
        <h2 className="text-2xl font-bold">{value}</h2>
        {change && (
          <span
            className={cn(
              "text-xs font-medium",
              change.positive
                ? "text-green-500 dark:text-green-400"
                : "text-red-500 dark:text-red-400"
            )}
          >
            {change.positive ? "+" : "-"}
            {change.value}
          </span>
        )}
      </div>
      {footer && <div className="mt-4 text-xs text-muted-foreground">{footer}</div>}
    </div>
  );
}
