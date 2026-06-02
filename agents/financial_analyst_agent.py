"""Financial analyst agent with OpenAI tool-calling and Phoenix auto-tracing."""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from openai import OpenAI
from openinference.instrumentation.openai import OpenAIInstrumentor
from opentelemetry.trace import StatusCode
from phoenix.otel import register
from pydantic import BaseModel, Field

load_dotenv()

PROJECT_NAME = "CausalAgentBench"
RISK_FREE_RATE = 0.04
DEFAULT_MODEL = "gpt-4o-mini"

_tracing_enabled = False
_tracer: Any = None
_tracer_provider: Any = None
_run_context: dict[str, Any] = {}

PROMPT_VERSIONS: dict[str, str] = {
    "v1_brief": (
        "You are a financial analyst. Use the available tools to fetch stock data, "
        "calculate metrics, and generate a comparison summary."
    ),
    "v2_detailed": (
        "You are a senior financial analyst. Always call get_stock_data, then "
        "calculate_metrics, then generate_summary. Be precise and only use computed numbers."
    ),
    "v3_structured": (
        "You are a financial analyst. Follow this exact workflow:\n"
        "1) get_stock_data for all requested tickers\n"
        "2) calculate_metrics on the returned data\n"
        "3) generate_summary using those metrics\n"
        "Never invent numbers."
    ),
}


def setup_tracing(*, verbose: bool = False, endpoint: str | None = None) -> None:
    """
    Enable Phoenix tracing for OpenAI calls and manual tool/agent spans.

    Call after px.launch_app() in the notebook so spans export to the local Phoenix UI.
    """
    global _tracing_enabled, _tracer, _tracer_provider
    global get_stock_data, calculate_metrics, generate_summary_tool

    if _tracing_enabled:
        return

    register_kwargs: dict[str, Any] = {"project_name": PROJECT_NAME, "verbose": verbose}
    if endpoint:
        register_kwargs["endpoint"] = endpoint

    _tracer_provider = register(**register_kwargs)
    OpenAIInstrumentor().instrument(tracer_provider=_tracer_provider)
    _tracer = _tracer_provider.get_tracer(__name__)

    get_stock_data = _tracer.tool(name="get_stock_data")(_get_stock_data_impl)
    calculate_metrics = _tracer.tool(name="calculate_metrics")(_calculate_metrics_impl)
    generate_summary_tool = _tracer.tool(name="generate_summary")(_generate_summary_tool_impl)

    _tracing_enabled = True


class GetStockDataInput(BaseModel):
    tickers: list[str] = Field(description="Stock ticker symbols, e.g. ['AAPL', 'MSFT']")
    start_date: str = Field(description="Start date in YYYY-MM-DD format")
    end_date: str = Field(description="End date in YYYY-MM-DD format")


class CalculateMetricsInput(BaseModel):
    data: dict[str, list[dict[str, Any]]] = Field(
        description="Stock price data keyed by ticker from get_stock_data"
    )


class GenerateSummaryInput(BaseModel):
    metrics: dict[str, dict[str, float | str]] = Field(
        description="Computed metrics keyed by ticker from calculate_metrics"
    )


def _get_stock_data_impl(
    tickers: list[str], start_date: str, end_date: str
) -> dict[str, list[dict[str, Any]]]:
    """Fetch daily close prices for each ticker via yfinance."""
    result: dict[str, list[dict[str, Any]]] = {}
    for ticker in tickers:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
        if df.empty:
            result[ticker] = []
            continue

        close = df["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]

        result[ticker] = [
            {"date": str(idx.date()), "close": float(val)}
            for idx, val in close.dropna().items()
        ]
    return result


def _extract_close_series(prices: Any) -> pd.Series:
    """
    Normalize price data to a float Series.

    Handles tool output ({date, close} dicts) and common LLM-simplified formats
    (plain float lists, or {closes: [...]}).
    """
    if prices is None:
        return pd.Series(dtype=float)

    if isinstance(prices, pd.Series):
        return prices.astype(float)

    if isinstance(prices, dict):
        if "close" in prices:
            close_val = prices["close"]
            if isinstance(close_val, list):
                return pd.Series([float(x) for x in close_val], dtype=float)
            if isinstance(close_val, (int, float)):
                return pd.Series([float(close_val)], dtype=float)
        if "closes" in prices and isinstance(prices["closes"], list):
            return pd.Series([float(x) for x in prices["closes"]], dtype=float)
        values = list(prices.values())
        if values and all(isinstance(v, (int, float)) for v in values):
            return pd.Series([float(v) for v in values], dtype=float)
        if values and all(isinstance(v, dict) for v in values):
            return _extract_close_series(values)

    if isinstance(prices, list):
        if not prices:
            return pd.Series(dtype=float)
        first = prices[0]
        if isinstance(first, dict):
            closes: list[float] = []
            for item in prices:
                if isinstance(item, dict) and "close" in item:
                    closes.append(float(item["close"]))
                elif isinstance(item, (int, float)):
                    closes.append(float(item))
            return pd.Series(closes, dtype=float)
        if isinstance(first, (int, float)):
            return pd.Series([float(x) for x in prices], dtype=float)

    return pd.Series(dtype=float)


def _calculate_metrics_impl(
    data: dict[str, Any],
) -> dict[str, dict[str, float | str]]:
    """Compute cumulative return, annualized return, annualized std dev, and Sharpe ratio."""
    metrics: dict[str, dict[str, float | str]] = {}
    for ticker, prices in data.items():
        closes = _extract_close_series(prices)
        if len(closes) < 2:
            metrics[ticker] = {"error": "insufficient data"}
            continue

        daily_returns = closes.pct_change().dropna()
        cumulative_return = float((closes.iloc[-1] / closes.iloc[0]) - 1)
        trading_days = len(daily_returns)
        years = trading_days / 252 if trading_days else 0
        annualized_return = float((1 + cumulative_return) ** (1 / years) - 1) if years > 0 else 0.0
        annualized_std = float(daily_returns.std() * np.sqrt(252))
        sharpe = (
            float((annualized_return - RISK_FREE_RATE) / annualized_std)
            if annualized_std > 0
            else 0.0
        )

        metrics[ticker] = {
            "cumulative_return": round(cumulative_return, 4),
            "annualized_return": round(annualized_return, 4),
            "annualized_std": round(annualized_std, 4),
            "sharpe_ratio": round(sharpe, 4),
        }
    return metrics


def _generate_summary_impl(
    metrics: dict[str, dict[str, float | str]],
    *,
    client: OpenAI,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
) -> str:
    """Call GPT to write a brief comparison narrative grounded in computed metrics."""
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a financial analyst. Write a brief comparison narrative "
                    "using only the provided metrics. Do not invent numbers."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Metrics:\n{json.dumps(metrics, indent=2)}\n\n"
                    "Write a brief comparison of these stocks."
                ),
            },
        ],
    )
    return response.choices[0].message.content or ""


def _generate_summary_tool_impl(metrics: dict[str, dict[str, float | str]]) -> str:
    return _generate_summary_impl(
        metrics,
        client=_run_context["client"],
        model=_run_context["model"],
        temperature=_run_context["temperature"],
    )


get_stock_data = _get_stock_data_impl
calculate_metrics = _calculate_metrics_impl
generate_summary_tool = _generate_summary_tool_impl


def _pydantic_to_openai_tool(model: type[BaseModel], name: str, description: str) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": model.model_json_schema(),
        },
    }


TOOLS = [
    _pydantic_to_openai_tool(
        GetStockDataInput,
        "get_stock_data",
        "Fetch daily stock close prices for tickers over a date range.",
    ),
    _pydantic_to_openai_tool(
        CalculateMetricsInput,
        "calculate_metrics",
        "Compute cumulative return, annualized return, annualized std dev, and Sharpe ratio.",
    ),
    _pydantic_to_openai_tool(
        GenerateSummaryInput,
        "generate_summary",
        "Generate a brief comparison narrative from computed metrics.",
    ),
]


def _estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Rough per-run cost estimate for common OpenAI models."""
    # source: https://holori.com/openai-pricing-guide/
    pricing = {
        "gpt-4o": (2.50 / 1_000_000, 10.00 / 1_000_000),
        "gpt-4o-mini": (0.15 / 1_000_000, 0.60 / 1_000_000),
    }
    input_rate, output_rate = pricing.get(model, pricing["gpt-4o-mini"])
    return round(input_tokens * input_rate + output_tokens * output_rate, 6)


def _handle_tool_calls( # function is a private helper
    tool_calls: list[Any],
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str], dict[str, dict[str, float | str]] | None, str]:
    tool_calls_made: list[str] = []
    metrics: dict[str, dict[str, float | str]] | None = None
    summary = ""

    for tool_call in tool_calls:
        name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)
        tool_calls_made.append(name)

        if name == "get_stock_data":
            result = get_stock_data(**args)
            _run_context["stock_data"] = result
        elif name == "calculate_metrics":
            # Use cached fetch output — the LLM often passes truncated/simplified data
            data = _run_context.get("stock_data") or args.get("data", {})
            result = calculate_metrics(data)
        elif name == "generate_summary":
            result = generate_summary_tool(**args)
        else:
            raise ValueError(f"Unknown tool: {name}")
        if name == "calculate_metrics":
            metrics = result
        elif name == "generate_summary":
            summary = result

        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result if name != "generate_summary" else {"summary": result}),
            }
        )

    return messages, tool_calls_made, metrics, summary


def run_agent(
    query: str,
    *,
    prompt_version: str = "v1_brief",
    temperature: float = 0.0,
    model: str = DEFAULT_MODEL,
    max_turns: int = 10,
) -> dict[str, Any]:
    """
    Run the financial analyst agent with OpenAI tool-calling.

    Returns a dict with summary, metrics, tool_calls_made, token usage, and cost.
    Phoenix traces LLM calls automatically and tool/agent spans when setup_tracing() ran.
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    system_prompt = PROMPT_VERSIONS.get(prompt_version, PROMPT_VERSIONS["v1_brief"])

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]

    _run_context.update(
        {
            "client": client,
            "model": model,
            "temperature": temperature,
            "stock_data": None,
        }
    )

    tool_calls_made: list[str] = []
    input_tokens = 0
    output_tokens = 0
    metrics: dict[str, dict[str, float | str]] | None = None
    summary = ""

    def _run_agent_loop() -> dict[str, Any]:
        nonlocal messages, tool_calls_made, input_tokens, output_tokens, metrics, summary

        for _ in range(max_turns):
            if _tracer:
                with _tracer.start_as_current_span(
                    "router_call", openinference_span_kind="chain"
                ) as router_span:
                    router_span.set_input(value=messages)
                    response = client.chat.completions.create(
                        model=model,
                        temperature=temperature,
                        messages=messages,
                        tools=TOOLS,
                        tool_choice="auto",
                    )
                    message = response.choices[0].message
                    router_span.set_output(
                        value=message.tool_calls or message.content
                    )
                    router_span.set_status(StatusCode.OK)
            else:
                response = client.chat.completions.create(
                    model=model,
                    temperature=temperature,
                    messages=messages,
                    tools=TOOLS,
                    tool_choice="auto",
                )
                message = response.choices[0].message

            usage = response.usage
            if usage:
                input_tokens += usage.prompt_tokens or 0
                output_tokens += usage.completion_tokens or 0

            messages.append(message.model_dump(exclude_none=True))

            if not message.tool_calls:
                summary = message.content or summary
                break

            if _tracer:
                with _tracer.start_as_current_span(
                    "handle_tool_calls", openinference_span_kind="chain"
                ) as tool_span:
                    tool_span.set_input(
                        value=[tool_call.function.name for tool_call in message.tool_calls]
                    )
                    messages, turn_tools, metrics, summary = _handle_tool_calls(
                        message.tool_calls, messages
                    )
                    tool_calls_made.extend(turn_tools)
                    tool_span.set_output(value={"tools": turn_tools, "summary": summary})
                    tool_span.set_status(StatusCode.OK)
            else:
                messages, turn_tools, metrics, summary = _handle_tool_calls(
                    message.tool_calls, messages
                )
                tool_calls_made.extend(turn_tools)
        else:
            summary = summary or "Agent did not finish within max_turns."

        required_tools = {"get_stock_data", "calculate_metrics", "generate_summary"}
        task_success = required_tools.issubset(set(tool_calls_made)) and bool(summary)

        return {
            "summary": summary,
            "metrics": metrics or {},
            "tool_calls_made": tool_calls_made,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": _estimate_cost_usd(model, input_tokens, output_tokens),
            "task_success": task_success,
            "prompt_version": prompt_version,
            "temperature": temperature,
            "model": model,
        }

    agent_input = {
        "query": query,
        "prompt_version": prompt_version,
        "temperature": temperature,
        "model": model,
    }

    if _tracer:
        with _tracer.start_as_current_span("AgentRun", openinference_span_kind="agent") as agent_span:
            agent_span.set_input(value=agent_input)
            result = _run_agent_loop()
            agent_span.set_output(value=result)
            agent_span.set_status(StatusCode.OK)
            return result

    return _run_agent_loop()


if __name__ == "__main__":
    setup_tracing(verbose=False)
    result = run_agent(
        "Compare AAPL and MSFT from 2023-01-01 to 2023-12-31.",
        prompt_version="v1_brief",
        temperature=0.0,
    )
    print(json.dumps(result, indent=2))
