import json
import datetime
from typing import Dict, Any

class HTMLGenerator:
    """
    Generates a standalone HTML report from debug analysis results.
    """
    
    def generate(self, results: Dict[str, Any], output_path: str):
        """
        Args:
            results: Combined dictionary from all analyzers.
            output_path: File path to save HTML.
        """
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Calculate health score (0-100)
        # Deduct points for issues
        health_score = 100
        issues = results.get("all_issues", [])
        for issue in issues:
            if issue['severity'] == 'critical': health_score -= 20
            elif issue['severity'] == 'warning': health_score -= 5
            elif issue['severity'] == 'info': health_score -= 1
        health_score = max(0, health_score)
        
        health_color = "green"
        if health_score < 80: health_color = "orange"
        if health_score < 50: health_color = "red"
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Transformer Debug Report</title>
            <style>
                body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 0; background: #f5f5f5; color: #333; }}
                .container {{ max_width: 1200px; margin: 20px auto; padding: 20px; }}
                .header {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); display: flex; justify-content: space-between; align-items: center; }}
                .score-card {{ text-align: center; }}
                .score {{ font-size: 48px; font-weight: bold; color: {health_color}; }}
                .card {{ background: white; margin-top: 20px; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                h2 {{ margin-top: 0; color: #2c3e50; border-bottom: 2px solid #eee; padding-bottom: 10px; }}
                .issue {{ padding: 10px; margin: 5px 0; border-radius: 4px; display: flex; align-items: center; }}
                .critical {{ background: #ffebee; border-left: 4px solid #ef5350; color: #c62828; }}
                .warning {{ background: #fff3e0; border-left: 4px solid #ffa726; color: #ef6c00; }}
                .info {{ background: #e3f2fd; border-left: 4px solid #42a5f5; color: #1565c0; }}
                .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 15px; }}
                .metric {{ background: #f8f9fa; pading: 15px; border-radius: 6px; text-align: center; }}
                .metric-val {{ font-size: 24px; font-weight: bold; }}
                .metric-label {{ font-size: 14px; color: #666; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <div>
                        <h1>Transformer Debug Report</h1>
                        <p>Generated: {timestamp}</p>
                    </div>
                    <div class="score-card">
                        <div class="score">{health_score}</div>
                        <div>Health Score</div>
                    </div>
                </div>
                
                <div class="card">
                    <h2>⚠️ Issues Detected ({len(issues)})</h2>
                    <div id="issues-list">
        """
        
        if not issues:
            html += "<p>✅ No issues detected. Model looks healthy!</p>"
        
        for issue in issues:
            html += f"""
            <div class="issue {issue['severity']}">
                <strong>[{issue['severity'].upper()}]</strong>&nbsp; {issue['msg']}
            </div>
            """
            
        html += """
                    </div>
                </div>
                
                <div class="card">
                    <h2>📊 Diagnostics Overview</h2>
                    <div class="metric-grid">
        """
        
        # summary stats
        hallucination = results.get("hallucination", {}).get("risk_score", 0)
        dead_neurons = results.get("weights", {}).get("summary", {}).get("dead_neurons", "N/A")
        
        html += self._metric_html("Hallucination Risk", f"{hallucination:.1f}", color="red" if hallucination > 50 else "black")
        html += self._metric_html("Dead Neurons", dead_neurons)
        
        if "hallucination" in results:
             metrics = results["hallucination"].get("metrics", {})
             html += self._metric_html("Output Entropy", f"{metrics.get('avg_entropy', 0):.2f}")
             html += self._metric_html("Confidence", f"{metrics.get('avg_confidence', 0):.2f}")
             
        html += """
                    </div>
                </div>
                
                <div class="card">
                    <h2>Raw Results (JSON)</h2>
                    <pre style="background: #eee; padding: 10px; overflow-x: auto;">
""" + json.dumps(results, indent=2) + """
                    </pre>
                </div>
            </div>
        </body>
        </html>
        """
        
        with open(output_path, "w", encoding='utf-8') as f:
            f.write(html)
        print(f"Report generated at {output_path}")

    def _metric_html(self, label, value, color="black"):
        return f"""
        <div class="metric" style="padding:10px;">
            <div class="metric-val" style="color:{color}">{value}</div>
            <div class="metric-label">{label}</div>
        </div>
        """
