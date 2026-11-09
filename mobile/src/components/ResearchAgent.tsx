import { useEffect, useRef, useState } from 'react';
import { Bot, ArrowUpRight, FileText, LoaderCircle, CheckCircle2 } from 'lucide-react';
import { askAgent, runAgentDemo, type AgentResult, type AgentSource } from '../lib/agent';

export default function ResearchAgent({ question, offline, url, token, disabled, onSource }: { question: string; offline: boolean; url: string; token: string; disabled: boolean; onSource: (source: AgentSource, result: AgentResult) => void }) {
  const operation = useRef(0);
  const [result, setResult] = useState<AgentResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  useEffect(() => { operation.current++; setResult(null); setError(''); setBusy(false); return () => { operation.current++; }; }, [question, offline, url]);
  async function run() {
    setResult(null); setError('');
    const clean = question.trim();
    if (!clean || clean.length > 1000) { setError('Ask an agent question between 1 and 1,000 characters.'); return; }
    if (!offline && !url.trim()) { setError('Add a research agent URL in Connections first.'); return; }
    const id = ++operation.current; setBusy(true);
    try { const answer = offline ? await runAgentDemo(clean) : await askAgent(url, clean, token); if (id === operation.current) setResult(answer); }
    catch (e) { if (id === operation.current) setError(e instanceof Error ? e.message : 'The research agent is unavailable.'); }
    finally { if (id === operation.current) setBusy(false); }
  }
  return <section className="agent-panel" aria-label="Research agent">
    <div className="agent-heading"><Bot size={23} /><div><h2>Ask your research agent</h2><p>{offline ? 'Scripted demo · search, inspect, cite' : 'Your agent service · read-only tools'}</p></div></div>
    <button className="secondary full" disabled={busy || disabled} onClick={() => void run()}>{busy ? <LoaderCircle size={18} className="spin" /> : <Bot size={18} />}{busy ? 'Following the evidence…' : 'Ask agent about this question'}<ArrowUpRight size={17} /></button>
    {error && <p className="error" role="alert">{error}</p>}
    {result && <div className="agent-result"><span className="eyebrow">{result.mode === 'demo' ? 'SCRIPTED AGENT DEMO' : 'CONNECTED RESEARCH AGENT'} · {result.status.replace('_', ' ')}</span><p className="agent-answer">{result.answer}</p><div className="agent-citations">{result.citations.map(c => <button key={c.source_id} onClick={() => onSource(result.sources.find(s => s.source_id === c.source_id)!, result)}><FileText size={14} />{c.source_id} · {c.path?.split('/').at(-1) ?? 'Source'}<ArrowUpRight size={13} /></button>)}</div><p className="helper"><CheckCircle2 size={14} /> References checked against inspected excerpts. Claim accuracy still needs review.</p><details className="technical"><summary>View research agent trace ({result.trace.length} tool calls)</summary><ol className="trace-list">{result.trace.map(step => <li key={step.step}><CheckCircle2 size={18} /><div><strong>{step.tool.replaceAll('_', ' ')}</strong><p>{step.status} · {step.source_ids.join(', ') || 'No sources'}</p></div></li>)}</ol><p className="helper">{result.notice}</p></details></div>}
  </section>;
}
