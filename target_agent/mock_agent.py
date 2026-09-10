"""Deterministic good and bad agents for end-to-end evaluation."""

from time import perf_counter

from harness.schemas import AgentExecutionResult, AgentTestCase
from harness.trajectory_tracer import TrajectoryTracer
from target_agent.tools import calculate_average, query_sales_db


class GoodMockAgent:
    name = "mock-good"

    def __init__(self) -> None:
        self.known_tools = {"query_sales_db", "calculate_average"}

    async def execute(self, test_case: AgentTestCase) -> AgentExecutionResult:
        started = perf_counter()
        tracer = TrajectoryTracer()

        query = tracer.record_tool_call(
            call_id="query-1",
            name="query_sales_db",
            arguments={"quarter": "Q3", "year": 2024},
        )
        sales = query_sales_db(quarter="Q3", year=2024)
        tracer.record_tool_result(
            query,
            result={"total_sales": sales.total_sales, "deals": sales.deals},
        )

        average_call = tracer.record_tool_call(
            call_id="average-1",
            name="calculate_average",
            arguments={"total_sales": sales.total_sales, "deals": sales.deals},
        )
        average = calculate_average(total_sales=sales.total_sales, deals=sales.deals)
        tracer.record_tool_result(average_call, result={"average_deal_size": average})

        answer = f"Total sales were $1.2M and average deal size was ${average / 1000:g}k."
        tracer.record_final_answer(answer)
        return AgentExecutionResult(
            test_id=test_case.test_id,
            agent=self.name,
            final_answer=answer,
            trajectory=tracer.steps,
            latency_ms=(perf_counter() - started) * 1000,
        )


class BadMockAgent:
    name = "mock-bad"

    def __init__(self) -> None:
        self.known_tools = {"query_sales_db", "calculate_average"}

    async def execute(self, test_case: AgentTestCase) -> AgentExecutionResult:
        started = perf_counter()
        tracer = TrajectoryTracer()

        for call_id in ("query-1", "query-2"):
            query = tracer.record_tool_call(
                call_id=call_id,
                name="query_sales_db",
                arguments={"quarter": "Q2", "year": 2024},
            )
            sales = query_sales_db(quarter="Q2", year=2024)
            tracer.record_tool_result(
                query,
                result={"total_sales": sales.total_sales, "deals": sales.deals},
            )

        invented = tracer.record_tool_call(
            call_id="invented-1",
            name="lookup_crm",
            arguments={"customer": "all"},
        )
        tracer.record_tool_result(invented, error="unknown tool: lookup_crm")

        answer = "Total sales were $1.4M and average deal size was $31k."
        tracer.record_final_answer(answer)
        return AgentExecutionResult(
            test_id=test_case.test_id,
            agent=self.name,
            final_answer=answer,
            trajectory=tracer.steps,
            latency_ms=(perf_counter() - started) * 1000,
        )
