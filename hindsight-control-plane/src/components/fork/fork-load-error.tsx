// fork: cost-accounting
// The load-error line shared by the cost and quota cards: the card's localized message, followed by
// the proxy's own error text in parentheses when it gave one.

export function ForkLoadError({ message, detail }: { message: string; detail: string | null }) {
  return (
    <p className="text-sm text-destructive">
      {message}
      {detail && <span className="font-mono"> ({detail})</span>}
    </p>
  );
}
