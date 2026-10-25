"""Format market summaries, agent reviews, risk evaluations, paper trades, and portfolio status."""

from __future__ import annotations

from crypto_research.config import Settings
from crypto_research.indicators import IndicatorSnapshot
from crypto_research.paper_trading import (
    PaperExecution,
    PaperTradeCandidate,
    PortfolioSnapshot,
    Position,
    VirtualAccount,
)
from crypto_research.research import DualAgentResult
from crypto_research.risk import RiskEvaluation


def format_market_summary(
    settings: Settings, snapshot: IndicatorSnapshot, candle_count: int
) -> str:
    """Render a human-readable research snapshot. Not a trade recommendation."""

    momentum_pct = snapshot.momentum_30d * 100
    return "\n".join(
        [
            "Market summary (educational, not a trade signal)",
            f"  Exchange:     {settings.exchange_id}",
            f"  Symbol:       {settings.symbol}",
            f"  Timeframe:    {settings.timeframe}",
            f"  Closed bars:  {candle_count}",
            f"  As of (UTC):  {snapshot.as_of.isoformat()}",
            f"  Close:        {snapshot.close:.8f}",
            f"  SMA 50:       {snapshot.sma_50:.8f}",
            f"  SMA 200:      {snapshot.sma_200:.8f}",
            f"  30d momentum: {momentum_pct:.2f}%",
            f"  Trend label:  {snapshot.trend}",
            "",
            "This output uses only closed candles. It is not financial advice,",
            "and it does not place or recommend live orders.",
        ]
    )


def format_dual_agent_result(result: DualAgentResult) -> str:
    """Render validated agent JSON. Not a live trade ticket."""

    proposal = result.proposal
    critique = result.critique
    evidence = "; ".join(proposal.evidence)
    risks = "; ".join(proposal.risks)
    missing = "; ".join(proposal.missing_data) or "(none listed)"
    weaknesses = "; ".join(critique.weaknesses)
    missing_evidence = "; ".join(critique.missing_evidence) or "(none listed)"
    return "\n".join(
        [
            "Scout proposal (educational, not an order)",
            f"  Action:       {proposal.action}",
            f"  Confidence:   {proposal.confidence:.2f} (not P(profit))",
            f"  Thesis:       {proposal.thesis}",
            f"  Evidence:     {evidence}",
            f"  Risks:        {risks}",
            f"  Missing data: {missing}",
            "",
            "Skeptic critique (approval is not execution)",
            f"  Verdict:      {critique.verdict}",
            f"  Agrees:       {critique.agrees_with_action}",
            f"  Confidence:   {critique.confidence:.2f} (not P(profit))",
            f"  Rationale:    {critique.rationale}",
            f"  Weaknesses:   {weaknesses}",
            f"  Missing ev.:  {missing_evidence}",
            "",
            "No orders were sent. AI agents will never override Python risk limits.",
        ]
    )


def format_risk_evaluation(evaluation: RiskEvaluation) -> str:
    """Render deterministic risk filter evaluation outcome."""

    status = "PASSED" if evaluation.passed else "REJECTED"
    lines = [
        f"Deterministic Risk Filters ({status})",
        f"  Proposed action:  {evaluation.proposed_action}",
        f"  Filters passed:   {evaluation.passed}",
        f"  Approved size:    {evaluation.approved_size_usd:.2f} USD",
    ]
    if evaluation.checks:
        lines.append("  Checks detail:")
        for check_name, passed in evaluation.checks.items():
            mark = "PASS" if passed else "FAIL"
            lines.append(f"    - {check_name}: {mark}")
    if evaluation.rejection_reasons:
        lines.append("  Rejection reasons:")
        for reason in evaluation.rejection_reasons:
            lines.append(f"    - {reason}")
    lines.append("")
    lines.append("Deterministic risk filters are authoritative; AI agents cannot override them.")
    return "\n".join(lines)


def format_paper_trade_candidate(candidate: PaperTradeCandidate) -> str:
    """Render simulated paper-trade candidate."""

    return "\n".join(
        [
            "Paper-trade candidate (SIMULATION ONLY - NO LIVE ORDERS)",
            f"  Candidate ID:    {candidate.candidate_id}",
            f"  Symbol:          {candidate.symbol}",
            f"  Action:          {candidate.action}",
            f"  Ref. price:      {candidate.reference_price:.8f}",
            f"  Size (USD):      {candidate.size_usd:.2f} USD",
            f"  Size (Asset):    {candidate.size_asset:.8f}",
            f"  As of (UTC):     {candidate.as_of_utc}",
            f"  Scout conf.:     {candidate.scout_confidence:.2f}",
            f"  Live trading:    DISABLED (is_live={candidate.is_live})",
            f"  Status:          {candidate.status}",
            "",
            "This is a paper candidate for audit and simulation only. Live orders are disabled.",
        ]
    )


def format_paper_execution(execution: PaperExecution) -> str:
    """Render simulated paper execution result."""

    return "\n".join(
        [
            "Simulated Paper Execution (SIMULATION ONLY - NO LIVE ORDERS)",
            f"  Execution ID:    {execution.execution_id}",
            f"  Candidate ID:    {execution.candidate_id}",
            f"  Symbol:          {execution.symbol}",
            f"  Action:          {execution.action}",
            f"  Signal candle:   {execution.signal_candle_utc}",
            f"  Executed at:     {execution.execution_timestamp_utc}",
            f"  Raw price:       {execution.raw_price:.8f}",
            f"  Slippage:        {execution.slippage_pct:.4%}",
            f"  Execution price: {execution.execution_price:.8f}",
            f"  Quantity:        {execution.quantity:.8f}",
            f"  Notional:        {execution.notional_usd:.2f} USD",
            f"  Fee rate:        {execution.fee_rate:.4%}",
            f"  Fee paid:        {execution.fee_usd:.2f} USD",
            f"  Net cash impact: {execution.net_cash_impact_usd:.2f} USD",
            f"  Realized P&L:    {execution.realized_pnl_usd:.2f} USD",
            f"  Post cash:       {execution.post_cash_usd:.2f} USD",
            f"  Post position:   {execution.post_quantity:.8f} @ avg {execution.post_avg_entry_price:.8f}",
            f"  Live trading:    DISABLED (is_live={execution.is_live})",
            "",
            "This execution was simulated locally. Live trading is strictly disabled.",
        ]
    )


def format_portfolio_status(
    account: VirtualAccount,
    positions: list[Position],
    snapshot: PortfolioSnapshot | None = None,
) -> str:
    """Render current virtual portfolio status."""

    lines = [
        "Virtual Portfolio Status (SIMULATION ONLY - NO LIVE ORDERS)",
        f"  Account ID:      {account.account_id}",
        f"  Available cash:  {account.cash_usd:.2f} USD",
        f"  Initial cash:    {account.initial_cash_usd:.2f} USD",
        f"  Realized P&L:    {account.realized_pnl_usd:.2f} USD",
        f"  Total fees paid: {account.total_fees_paid_usd:.2f} USD",
        f"  Peak equity:     {account.peak_equity_usd:.2f} USD",
    ]
    if snapshot is not None:
        lines.extend(
            [
                f"  Current equity:  {snapshot.total_equity_usd:.2f} USD",
                f"  Unrealized P&L:  {snapshot.unrealized_pnl_usd:.2f} USD",
                f"  Max drawdown:    {snapshot.drawdown_pct:.2%}",
            ]
        )
    active_positions = [p for p in positions if p.quantity > 0]
    if active_positions:
        lines.append("  Open Positions:")
        for p in active_positions:
            lines.append(
                f"    - {p.symbol}: {p.quantity:.8f} units @ avg entry {p.average_entry_price:.8f} "
                f"(cost basis: {p.cost_basis_usd:.2f} USD)"
            )
    else:
        lines.append("  Open Positions:  (none)")
    lines.append("")
    lines.append("Virtual portfolio state is persisted in SQLite. Live orders cannot be placed.")
    return "\n".join(lines)
