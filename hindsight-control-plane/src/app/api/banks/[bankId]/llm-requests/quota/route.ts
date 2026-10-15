// fork: cost-accounting
import { proxyLLMCost } from "@/lib/fork/llm-cost-proxy";

export async function GET(request: Request, { params }: { params: Promise<{ bankId: string }> }) {
  const { bankId } = await params;
  return proxyLLMCost(request, bankId, "/llm-requests/quota");
}
