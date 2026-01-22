"""
MCP Server implementation for CryptoSentinel.

Implements the Model Context Protocol server that exposes CryptoSentinel
tools to AI assistants.
"""

import json
import logging
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime

from .tools import (
    reddit_search,
    sentiment_aggregate,
    topic_clusters,
    classify_post,
    generate_brief,
    CryptoSentinelTools,
    Post,
    SentimentReport,
    TopicCluster,
    NarrativeBrief
)
from ..models.classifier import ClassificationResult


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ToolDefinition:
    """MCP tool definition."""
    name: str
    description: str
    parameters: Dict[str, Any]


@dataclass
class ToolResult:
    """Result from a tool invocation."""
    success: bool
    data: Any
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class CryptoSentinelMCPServer:
    """
    MCP Server for CryptoSentinel.
    
    Exposes CryptoSentinel analysis tools via the Model Context Protocol
    for integration with AI assistants.
    """
    
    def __init__(self):
        """Initialize the MCP server."""
        self.tools = CryptoSentinelTools()
        self._tool_registry: Dict[str, Callable] = {}
        self._register_tools()
        logger.info("CryptoSentinel MCP Server initialized")
    
    def _register_tools(self) -> None:
        """Register all available tools."""
        self._tool_registry = {
            "reddit_search": self._handle_reddit_search,
            "sentiment_aggregate": self._handle_sentiment_aggregate,
            "topic_clusters": self._handle_topic_clusters,
            "classify_post": self._handle_classify_post,
            "generate_brief": self._handle_generate_brief,
        }
    
    def get_tool_definitions(self) -> List[ToolDefinition]:
        """
        Get definitions of all available tools.
        
        Returns:
            List of ToolDefinition objects for MCP discovery.
        """
        return [
            ToolDefinition(
                name="reddit_search",
                description="Search Reddit posts from r/wallstreetbets and r/CryptoMoonShots with optional filters",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query string"
                        },
                        "subreddit": {
                            "type": "string",
                            "description": "Filter to subreddit: 'wallstreetbets' or 'CryptoMoonShots'",
                            "enum": ["wallstreetbets", "CryptoMoonShots"]
                        },
                        "time_range": {
                            "type": "string",
                            "description": "Time range filter",
                            "enum": ["day", "week", "month", "year"]
                        },
                        "min_score": {
                            "type": "integer",
                            "description": "Minimum Reddit score",
                            "default": 0
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum results to return",
                            "default": 10
                        }
                    },
                    "required": ["query"]
                }
            ),
            ToolDefinition(
                name="sentiment_aggregate",
                description="Aggregate sentiment for a topic across Reddit posts",
                parameters={
                    "type": "object",
                    "properties": {
                        "topic": {
                            "type": "string",
                            "description": "Topic to analyze sentiment for"
                        },
                        "subreddit": {
                            "type": "string",
                            "description": "Filter to specific subreddit",
                            "enum": ["wallstreetbets", "CryptoMoonShots"]
                        }
                    },
                    "required": ["topic"]
                }
            ),
            ToolDefinition(
                name="topic_clusters",
                description="Extract main topic clusters from a subreddit",
                parameters={
                    "type": "object",
                    "properties": {
                        "subreddit": {
                            "type": "string",
                            "description": "Subreddit to analyze",
                            "enum": ["wallstreetbets", "CryptoMoonShots"]
                        },
                        "n_clusters": {
                            "type": "integer",
                            "description": "Number of clusters to extract",
                            "default": 5
                        }
                    },
                    "required": []
                }
            ),
            ToolDefinition(
                name="classify_post",
                description="Classify if a post is crypto or stocks related",
                parameters={
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "Post title"
                        },
                        "text": {
                            "type": "string",
                            "description": "Post body text (optional)"
                        }
                    },
                    "required": ["title"]
                }
            ),
            ToolDefinition(
                name="generate_brief",
                description="Generate a grounded narrative brief with citations about a topic",
                parameters={
                    "type": "object",
                    "properties": {
                        "topic": {
                            "type": "string",
                            "description": "Topic to generate brief about"
                        },
                        "max_sources": {
                            "type": "integer",
                            "description": "Maximum number of sources to include",
                            "default": 5
                        }
                    },
                    "required": ["topic"]
                }
            )
        ]
    
    def invoke_tool(self, name: str, arguments: Dict[str, Any]) -> ToolResult:
        """
        Invoke a tool by name with given arguments.
        
        Args:
            name: Tool name.
            arguments: Tool arguments.
            
        Returns:
            ToolResult with success status and data or error.
        """
        if name not in self._tool_registry:
            return ToolResult(
                success=False,
                data=None,
                error=f"Unknown tool: {name}"
            )
        
        try:
            handler = self._tool_registry[name]
            result = handler(arguments)
            return ToolResult(
                success=True,
                data=result,
                metadata={"tool": name, "timestamp": datetime.now().isoformat()}
            )
        except Exception as e:
            logger.error(f"Tool invocation failed: {name} - {e}")
            return ToolResult(
                success=False,
                data=None,
                error=str(e)
            )
    
    def _handle_reddit_search(self, args: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Handle reddit_search tool invocation."""
        posts = reddit_search(
            query=args["query"],
            subreddit=args.get("subreddit"),
            time_range=args.get("time_range"),
            min_score=args.get("min_score", 0),
            max_results=args.get("max_results", 10),
            tools=self.tools
        )
        return [asdict(p) for p in posts]
    
    def _handle_sentiment_aggregate(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle sentiment_aggregate tool invocation."""
        report = sentiment_aggregate(
            topic=args["topic"],
            subreddit=args.get("subreddit"),
            tools=self.tools
        )
        return {
            "topic": report.topic,
            "subreddit": report.subreddit,
            "total_posts": report.total_posts,
            "positive_count": report.positive_count,
            "negative_count": report.negative_count,
            "neutral_count": report.neutral_count,
            "average_sentiment": round(report.average_sentiment, 3),
            "sentiment_label": report.sentiment_label,
            "top_positive": [asdict(p) for p in report.top_positive],
            "top_negative": [asdict(p) for p in report.top_negative]
        }
    
    def _handle_topic_clusters(self, args: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Handle topic_clusters tool invocation."""
        clusters = topic_clusters(
            subreddit=args.get("subreddit"),
            n_clusters=args.get("n_clusters", 5),
            tools=self.tools
        )
        return [
            {
                "cluster_id": c.cluster_id,
                "label": c.label,
                "keywords": c.keywords,
                "post_count": c.post_count,
                "avg_sentiment": round(c.avg_sentiment, 3),
                "representative_posts": [asdict(p) for p in c.representative_posts]
            }
            for c in clusters
        ]
    
    def _handle_classify_post(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle classify_post tool invocation."""
        result = classify_post(
            title=args["title"],
            text=args.get("text"),
            tools=self.tools
        )
        return result.to_dict()
    
    def _handle_generate_brief(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle generate_brief tool invocation."""
        brief = generate_brief(
            topic=args["topic"],
            max_sources=args.get("max_sources", 5),
            tools=self.tools
        )
        return {
            "topic": brief.topic,
            "summary": brief.summary,
            "key_insights": brief.key_insights,
            "sentiment_overview": brief.sentiment_overview,
            "citations": [c.to_dict() for c in brief.citations],
            "confidence": round(brief.confidence, 3),
            "groundedness": brief.groundedness,
            "warnings": brief.warnings
        }
    
    def handle_jsonrpc(self, request: str) -> str:
        """
        Handle a JSON-RPC request.
        
        Args:
            request: JSON-RPC request string.
            
        Returns:
            JSON-RPC response string.
        """
        try:
            req = json.loads(request)
        except json.JSONDecodeError:
            return json.dumps({
                "jsonrpc": "2.0",
                "error": {"code": -32700, "message": "Parse error"},
                "id": None
            })
        
        req_id = req.get("id")
        method = req.get("method", "")
        params = req.get("params", {})
        
        if method == "tools/list":
            tools = self.get_tool_definitions()
            return json.dumps({
                "jsonrpc": "2.0",
                "result": {"tools": [asdict(t) for t in tools]},
                "id": req_id
            })
        
        elif method == "tools/call":
            tool_name = params.get("name")
            arguments = params.get("arguments", {})
            
            result = self.invoke_tool(tool_name, arguments)
            
            if result.success:
                return json.dumps({
                    "jsonrpc": "2.0",
                    "result": {"content": [{"type": "text", "text": json.dumps(result.data, indent=2)}]},
                    "id": req_id
                })
            else:
                return json.dumps({
                    "jsonrpc": "2.0",
                    "error": {"code": -32000, "message": result.error},
                    "id": req_id
                })
        
        else:
            return json.dumps({
                "jsonrpc": "2.0",
                "error": {"code": -32601, "message": f"Method not found: {method}"},
                "id": req_id
            })
    
    def run_stdio(self) -> None:
        """Run the server using stdio transport."""
        import sys
        
        logger.info("Starting CryptoSentinel MCP Server (stdio mode)")
        
        while True:
            try:
                line = sys.stdin.readline()
                if not line:
                    break
                
                response = self.handle_jsonrpc(line.strip())
                sys.stdout.write(response + "\n")
                sys.stdout.flush()
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error processing request: {e}")
                error_response = json.dumps({
                    "jsonrpc": "2.0",
                    "error": {"code": -32603, "message": str(e)},
                    "id": None
                })
                sys.stdout.write(error_response + "\n")
                sys.stdout.flush()


def main():
    """Entry point for running the MCP server."""
    server = CryptoSentinelMCPServer()
    server.run_stdio()


if __name__ == "__main__":
    main()
