from __future__ import annotations

import html
import json
from typing import Any

from .syllabus_manager import get_task_subject_syllabus
from .task_schema import LearningTask
from .task_store import connect_db


STATUS_LABELS = {
    "unvisited": "未接触",
    "seen": "已接触",
    "learning": "学习中",
    "weak": "薄弱",
    "mastered": "基本掌握",
    "proficient": "熟练掌握",
    "mixed_weak": "存在薄弱点",
}

STATUS_COLORS = {
    "unvisited": "#6b7280",
    "seen": "#38bdf8",
    "learning": "#f59e0b",
    "weak": "#ef4444",
    "mastered": "#22c55e",
    "proficient": "#14b8a6",
    "mixed_weak": "#f97316",
}


def _state_map(task_id: str) -> dict[str, dict[str, Any]]:
    with connect_db() as conn:
        rows = conn.execute("SELECT * FROM knowledge_state WHERE task_id = ?", (task_id,)).fetchall()
    return {row["knowledge_node_id"]: dict(row) for row in rows}


def _question_map(task_id: str) -> dict[str, list[dict[str, Any]]]:
    with connect_db() as conn:
        rows = conn.execute(
            """
            SELECT id, timestamp, user_input, assistant_answer, knowledge_node_id, knowledge_path
            FROM questions
            WHERE task_id = ?
            ORDER BY created_at DESC
            LIMIT 200
            """,
            (task_id,),
        ).fetchall()
    result: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        item = dict(row)
        item["answer_summary"] = _summary(item.get("assistant_answer") or "")
        result.setdefault(item.get("knowledge_node_id") or "general", []).append(item)
    return result


def _mistake_map(task_id: str) -> dict[str, list[dict[str, Any]]]:
    with connect_db() as conn:
        rows = conn.execute(
            """
            SELECT id, subject, knowledge_node_id, original_question, mistake_reason, next_review_at
            FROM mistake_book
            WHERE task_id = ?
            ORDER BY created_at DESC
            LIMIT 200
            """,
            (task_id,),
        ).fetchall()
    result: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        item = dict(row)
        result.setdefault(item.get("knowledge_node_id") or "general", []).append(item)
    return result


def render_profile(task: LearningTask, subject: str, mode: str = "tree") -> str:
    syllabus = get_task_subject_syllabus(task, subject)
    if not syllabus:
        return _missing_syllabus_html(task, subject)
    states = _state_map(task.id)
    body = _render_tree_lines(syllabus.get("nodes") or [], states, 0, mode)
    return "<pre class='profile-tree'>" + html.escape("\n".join(body)) + "</pre>"


def render_profile_visual(task: LearningTask, subject: str, view_mode: str = "知识网络") -> str:
    syllabus = get_task_subject_syllabus(task, subject)
    if not syllabus:
        return _missing_syllabus_html(task, subject)

    states = _state_map(task.id)
    questions = _question_map(task.id)
    mistakes = _mistake_map(task.id)
    graph = _build_graph_payload(syllabus, states, questions, mistakes)
    if view_mode == "树状视图":
        return _wrap_profile_html(render_profile(task, subject, "tree"))

    chart_kind = "tree" if view_mode == "思维导图" else "graph"
    payload = json.dumps(graph, ensure_ascii=False)
    container_id = f"profile_chart_{abs(hash((task.id, subject, view_mode))) % 1000000}"
    fallback_html = _fallback_node_map(graph)
    return f"""
<div class="profile-shell">
  <div class="profile-chart" id="{container_id}">
    <div class="chart-fallback">{fallback_html}</div>
  </div>
  <aside class="profile-detail" id="{container_id}_detail">
    <h3>点击一个知识点</h3>
    <p>这里会显示掌握状态、历史提问、错题和复习建议。</p>
  </aside>
</div>
<script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
<script>
(function() {{
  const payload = {payload};
  const el = document.getElementById("{container_id}");
  const detail = document.getElementById("{container_id}_detail");
  if (!el || !window.echarts) {{
    if (detail) detail.innerHTML = "<h3>图谱加载失败</h3><p>请切换到树状视图。</p>";
    return;
  }}
  const chart = echarts.init(el, null, {{renderer: "canvas"}});
  const option = buildOption(payload, "{chart_kind}");
  chart.setOption(option);
  const fallback = el.querySelector(".chart-fallback");
  if (fallback) fallback.style.display = "none";
  setTimeout(function() {{ chart.resize(); }}, 200);
  setTimeout(function() {{ chart.resize(); }}, 800);
  chart.on("click", function(params) {{
    const item = payload.details[params.data.id || params.data.name] || params.data.detail;
    if (item) detail.innerHTML = renderDetail(item);
  }});
  window.addEventListener("resize", function() {{ chart.resize(); }});

  function buildOption(data, kind) {{
    if (kind === "tree") {{
      return {{
        backgroundColor: "transparent",
        tooltip: {{trigger: "item"}},
        series: [{{
          type: "tree",
          data: [data.tree],
          top: "4%",
          left: "8%",
          bottom: "4%",
          right: "22%",
          symbolSize: function(value, params) {{ return params.data.symbolSize || 12; }},
          label: {{color: "#e5e7eb", position: "left", verticalAlign: "middle", align: "right"}},
          leaves: {{label: {{position: "right", align: "left"}}}},
          lineStyle: {{color: "#475569"}},
          itemStyle: {{color: "#38bdf8", borderColor: "#0f172a", borderWidth: 1}},
          emphasis: {{focus: "descendant"}},
          expandAndCollapse: true,
          animationDuration: 350,
          animationDurationUpdate: 500
        }}]
      }};
    }}
    return {{
      backgroundColor: "transparent",
      tooltip: {{formatter: function(p) {{ return p.data.name || ""; }}}},
      series: [{{
        type: "graph",
        layout: "force",
        roam: true,
        draggable: true,
        data: data.nodes,
        links: data.links,
        force: {{repulsion: 260, edgeLength: [70, 150]}},
        label: {{show: true, color: "#e5e7eb", fontSize: 12}},
        lineStyle: {{color: "#64748b", width: 1.2, opacity: 0.85}},
        emphasis: {{focus: "adjacency"}}
      }}]
    }};
  }}
  function renderDetail(item) {{
    const q = (item.questions || []).map(function(x, idx) {{
      return `<div class="qa-card"><b>${{idx + 1}}. ${{escapeHtml(x.timestamp || "")}}</b><p><b>问题：</b>${{escapeHtml(x.user_input || "")}}</p><p><b>答案摘要：</b>${{escapeHtml(x.answer_summary || "")}}</p><details><summary>展开完整答案</summary><div>${{escapeHtml(x.assistant_answer || "")}}</div></details></div>`;
    }}).join("") || "<p>暂无历史提问。</p>";
    const m = (item.mistakes || []).map(function(x) {{
      return `<li>${{escapeHtml(x.mistake_reason || "错题")}} - 下次复习：${{escapeHtml(x.next_review_at || "-")}}</li>`;
    }}).join("") || "<li>暂无错题。</li>";
    return `<h3>${{escapeHtml(item.name)}}</h3>
      <p class="muted">${{escapeHtml(item.id)}} · ${{escapeHtml((item.path || []).join(" > "))}}</p>
      <div class="stat-grid">
        <span>状态<br><b>${{escapeHtml(item.status_label)}}</b></span>
        <span>seen<br><b>${{item.seen_count || 0}}</b></span>
        <span>weak<br><b>${{item.weak_count || 0}}</b></span>
        <span>mistake<br><b>${{item.mistake_count || 0}}</b></span>
        <span>mastery<br><b>${{Number(item.mastery_score || 0).toFixed(2)}}</b></span>
      </div>
      <p><b>最近学习：</b>${{escapeHtml(item.last_seen_at || "-")}}</p>
      <p><b>复习建议：</b>${{escapeHtml(item.review_advice || "")}}</p>
      <h4>历史提问</h4>${{q}}
      <h4>错题记录</h4><ul>${{m}}</ul>
      <div class="node-actions"><button>标记 weak</button><button>标记 mastered</button><button>标记 proficient</button><button>加入错题本</button></div>
      <p class="muted">按钮为前端占位；批量修正可在错题本/小测验区域完成。</p>`;
  }}
  function escapeHtml(s) {{
    return String(s || "").replace(/[&<>"']/g, function(c) {{
      return {{"&":"&amp;","<":"&lt;",">":"&gt;","\\"":"&quot;","'":"&#039;"}}[c];
    }});
  }}
}})();
</script>
"""


def _build_graph_payload(
    syllabus: dict[str, Any],
    states: dict[str, dict[str, Any]],
    questions: dict[str, list[dict[str, Any]]],
    mistakes: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    links: list[dict[str, str]] = []
    details: dict[str, dict[str, Any]] = {}
    root_id = syllabus.get("syllabus_id") or syllabus.get("subject") or "root"
    root = {"id": root_id, "name": syllabus.get("subject", "知识框架"), "children": []}

    def walk(raw_nodes: list[dict[str, Any]], parent_id: str, path: list[str]) -> list[dict[str, Any]]:
        children = []
        for raw in raw_nodes or []:
            node_id = str(raw.get("id") or raw.get("name"))
            name = str(raw.get("name") or node_id)
            child_path = path + [name]
            state = states.get(node_id) or {}
            status = _aggregate_status(raw, states) if not state else state.get("status", "unvisited")
            detail = _detail(node_id, name, child_path, status, state, questions.get(node_id, []), mistakes.get(node_id, []))
            size = 22 + min(28, int(state.get("seen_count") or 0) * 4 + len(questions.get(node_id, [])) * 2)
            nodes.append(
                {
                    "id": node_id,
                    "name": name,
                    "symbolSize": size,
                    "itemStyle": {"color": STATUS_COLORS.get(status, STATUS_COLORS["unvisited"])},
                    "detail": detail,
                }
            )
            links.append({"source": parent_id, "target": node_id})
            details[node_id] = detail
            children.append(
                {
                    "id": node_id,
                    "name": name,
                    "symbolSize": max(10, size / 2),
                    "itemStyle": {"color": STATUS_COLORS.get(status, STATUS_COLORS["unvisited"])},
                    "detail": detail,
                    "children": walk(raw.get("children") or [], node_id, child_path),
                }
            )
        return children

    root_detail = _detail(root_id, root["name"], [root["name"]], "seen", {}, [], [])
    nodes.append({"id": root_id, "name": root["name"], "symbolSize": 34, "itemStyle": {"color": "#60a5fa"}, "detail": root_detail})
    details[root_id] = root_detail
    root["children"] = walk(syllabus.get("nodes") or [], root_id, [root["name"]])
    return {"nodes": nodes, "links": links, "tree": root, "details": details}


def _detail(
    node_id: str,
    name: str,
    path: list[str],
    status: str,
    state: dict[str, Any],
    questions: list[dict[str, Any]],
    mistakes: list[dict[str, Any]],
) -> dict[str, Any]:
    weak = int(state.get("weak_count") or 0)
    mistake = int(state.get("mistake_count") or 0)
    advice = "建议先复习定义和典型例题。"
    if weak or mistake:
        advice = "这是当前薄弱点，建议复习定义、做 2-3 道基础题，并回看错题原因。"
    elif status in {"mastered", "proficient"}:
        advice = "保持间隔复习，可用小测验巩固熟练度。"
    return {
        "id": node_id,
        "name": name,
        "path": path,
        "status": status,
        "status_label": STATUS_LABELS.get(status, "未接触"),
        "seen_count": int(state.get("seen_count") or 0),
        "weak_count": weak,
        "mistake_count": mistake,
        "review_count": int(state.get("review_count") or 0),
        "mastery_score": float(state.get("mastery_score") or 0),
        "last_seen_at": state.get("last_seen_at") or "",
        "questions": questions,
        "mistakes": mistakes,
        "review_advice": advice,
    }


def _aggregate_status(node: dict[str, Any], states: dict[str, dict[str, Any]]) -> str:
    children = node.get("children") or []
    if not children:
        return "unvisited"
    child_statuses = []
    for child in children:
        child_id = str(child.get("id") or "")
        status = (states.get(child_id) or {}).get("status") or _aggregate_status(child, states)
        child_statuses.append(status)
    if any(status in {"weak", "mixed_weak"} for status in child_statuses):
        return "mixed_weak"
    if child_statuses and all(status in {"mastered", "proficient"} for status in child_statuses):
        return "mastered"
    if child_statuses and all(status != "unvisited" for status in child_statuses):
        return "seen"
    return "unvisited"


def _render_tree_lines(nodes: list[dict[str, Any]], states: dict[str, dict[str, Any]], level: int, mode: str) -> list[str]:
    lines: list[str] = []
    for node in nodes or []:
        node_id = str(node.get("id") or "")
        state = states.get(node_id)
        status = (state or {}).get("status") or _aggregate_status(node, states)
        include = mode != "gaps" or status in {"unvisited", "weak", "learning", "mixed_weak"}
        child_lines = _render_tree_lines(node.get("children") or [], states, level + 1, mode)
        if include or child_lines:
            indent = "  " * level
            label = STATUS_LABELS.get(status, "未接触")
            stats = ""
            if state:
                stats = f" seen:{state.get('seen_count', 0)} weak:{state.get('weak_count', 0)} mistake:{state.get('mistake_count', 0)} mastery:{float(state.get('mastery_score') or 0):.2f}"
            lines.append(f"{indent}- [{label}] {node.get('name')}{stats}")
            lines.extend(child_lines)
    return lines


def _missing_syllabus_html(task: LearningTask, subject: str) -> str:
    return f"""
<div class="empty-syllabus">
  <h3>当前任务尚未绑定「{html.escape(subject or '当前科目')}」的知识框架</h3>
  <p>你可以在任务管理中补充知识框架，或选择该任务已绑定的其他科目。</p>
  <ol>
    <li>切到「任务管理」</li>
    <li>选择「自定义」角色或补充知识框架文本</li>
    <li>重新创建/绑定任务后刷新画像</li>
  </ol>
  <p class="muted">任务：{html.escape(task.task_name or task.id)} / {html.escape(task.role_type)}</p>
</div>
"""


def _wrap_profile_html(content: str) -> str:
    return f"<div class='profile-shell profile-shell-single'>{content}</div>"


def _fallback_node_map(graph: dict[str, Any]) -> str:
    items = []
    for node in graph.get("nodes", []):
        detail = node.get("detail") or {}
        color = (node.get("itemStyle") or {}).get("color", "#64748b")
        label = html.escape(detail.get("status_label") or "")
        name = html.escape(node.get("name") or "")
        path = html.escape(" > ".join(detail.get("path") or []))
        items.append(
            f"<div class='fallback-node' style='border-left-color:{color}'>"
            f"<b>{name}</b><span>{label}</span><small>{path}</small></div>"
        )
    return "<div class='fallback-grid'>" + "".join(items) + "</div>"


def _summary(text: str, limit: int = 180) -> str:
    clean = " ".join((text or "").split())
    return clean[:limit] + ("..." if len(clean) > limit else "")
