"""Report generation agent: compiles all results into structured output."""

from agesensei.schema import DiscoveryReport


class ReporterAgent:
    """Generate final discovery report with LLM-powered executive summary.

    Output formats:
        - Markdown report
        - HTML report
        - Streamlit interactive dashboard

    Example:
        agent = ReporterAgent(llm=claude)
        agent.generate_markdown(report, "output/report.md")
        agent.launch_dashboard(report)
    """

    def __init__(self, llm=None):
        self.llm = llm

    async def generate_summary(self, report: DiscoveryReport) -> str:
        """LLM-generated executive summary of the discovery results."""
        raise NotImplementedError

    def generate_markdown(self, report: DiscoveryReport, output_path: str):
        """Generate a Markdown report file."""
        raise NotImplementedError

    def generate_html(self, report: DiscoveryReport, output_path: str):
        """Generate an HTML report with embedded visualizations."""
        raise NotImplementedError

    def launch_dashboard(self, report: DiscoveryReport, port: int = 8501):
        """Launch Streamlit interactive dashboard."""
        raise NotImplementedError

    def _render_target_table(self, report: DiscoveryReport) -> str:
        """Render ranked target table."""
        raise NotImplementedError

    def _render_pathway_network(self, report: DiscoveryReport) -> str:
        """Render pathway-target network visualization."""
        raise NotImplementedError
