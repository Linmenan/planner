# PlanScope

This is the official repository of

**PlanScope: Learning to Plan Within Decision Scope Does Matter**

[Ren Xin](https://rex-sys-hk.github.io), [Jie Cheng](https://jchengai.github.io/), [Hongji Liu](http://liuhongji.site) and [Jun Ma](https://personal.hkust-gz.edu.cn/junma/index.html)


<p align="left">
<a href="https://rex-sys-hk.github.io/pub_webs/PlanScope/">
<img src="https://img.shields.io/badge/Project-Page-blue?style=flat">
</a>
<a href='https://arxiv.org/abs/2411.00476' style='padding-left: 0.5rem;'>
    <img src='https://img.shields.io/badge/arXiv-PDF-red?style=flat&logo=arXiv&logoColor=wihte' alt='arXiv PDF'>
</a>
</p>

![PlanScopeConcept](https://github.com/user-attachments/assets/c622cb18-8ebe-4b70-94c7-6d7a4c443260)
import torch
import plotly.graph_objects as go
from plotly.colors import qualitative
import ipywidgets as widgets
from IPython.display import display

# -------------------- 数据加载 --------------------
# 加载保存的 pt 文件
map_data = torch.load("./map_data.pt")
map_mask_data = torch.load("./map_mask_data.pt")
agent_position_data = torch.load("./agent_position_data.pt")
agent_mask_data = torch.load("./agent_mask_data.pt")

# -------------------- 参数设置 --------------------
# 定义地图车道颜色组，每组包含中心线、左边界和右边界的颜色
color_pairs = [
    {"center": "#2C3E50", "left": "#1f77b4", "right": "#d62728"},  # 组0：中心线带蓝色调
    {"center": "#34495E", "left": "#17becf", "right": "#ff7f0e"},  # 组1：中心线带蓝灰调
    {"center": "#8E44AD", "left": "#0074D9", "right": "#FF4136"},  # 组2：中心线带紫色调
    {"center": "#16A085", "left": "#2E86C1", "right": "#E74C3C"},  # 组3：中心线带青绿色调
    {"center": "#27AE60", "left": "#2980B9", "right": "#E67E22"},  # 组4：中心线带绿色调
    {"center": "#D35400", "left": "#3498DB", "right": "#F39C12"},  # 组5：中心线带橙色调
    {"center": "#C0392B", "left": "#5DADE2", "right": "#E74C3C"},  # 组6：中心线带红色调
]

# 计算所有帧中 agent 数量的最大值（agent 按 A 维度排序，即相同位置即为同一目标）
max_A = max([agent_position_item.shape[0] for agent_position_item in agent_position_data])
# 使用 Plotly 的定性配色方案作为 agent 的颜色池（不足时循环使用）
agent_colors_pool = qualitative.Plotly  
agent_colors = [agent_colors_pool[i % len(agent_colors_pool)] for i in range(max_A)]
n_frames = len(map_data)

# -------------------- 构建动画图的函数 --------------------
def build_figure(select_agent_id):
    """
    根据所选 agent id（select_agent_id）构建动画图，
    仅显示该 agent（位置为 select_agent_id），其它 agent 添加空 trace，
    保证每一帧中 agent trace 的顺序和数量一致，从而使得颜色稳定。
    """
    frames = []
    
    for frame_idx, (map_item, map_mask_item, agent_position_item, agent_mask_item) in enumerate(
            zip(map_data, map_mask_data, agent_position_data, agent_mask_data)):
        
        traces = []
        # 地图数据：假设 map_item.shape 为 [D1, D2, P, 2]，其中 D1 为车道数量
        D1, _, _, _ = map_item.shape
        for lane_idx in range(D1):
            color_set = color_pairs[lane_idx % len(color_pairs)]
            lane_mask = map_mask_item[lane_idx]  # 布尔 mask, shape: [P]
            lines = map_item[lane_idx]           # shape: [D2, P, 2]
            center_line = lines[0]
            left_bound = lines[1]
            right_bound = lines[2]
            # 筛选有效点
            center_valid = center_line[lane_mask]
            left_valid = left_bound[lane_mask]
            right_valid = right_bound[lane_mask]
            
            # 始终添加三个 trace（数据为空时 x,y 为 []）
            traces.append(go.Scatter(
                x=center_valid[:, 0].tolist() if center_valid.shape[0] > 0 else [],
                y=center_valid[:, 1].tolist() if center_valid.shape[0] > 0 else [],
                mode='lines',
                name=f"Lane {lane_idx} 中心线",
                line=dict(color=color_set["center"], dash="dot")
            ))
            traces.append(go.Scatter(
                x=left_valid[:, 0].tolist() if left_valid.shape[0] > 0 else [],
                y=left_valid[:, 1].tolist() if left_valid.shape[0] > 0 else [],
                mode='lines',
                name=f"Lane {lane_idx} 左边界",
                line=dict(color=color_set["left"])
            ))
            traces.append(go.Scatter(
                x=right_valid[:, 0].tolist() if right_valid.shape[0] > 0 else [],
                y=right_valid[:, 1].tolist() if right_valid.shape[0] > 0 else [],
                mode='lines',
                name=f"Lane {lane_idx} 右边界",
                line=dict(color=color_set["right"])
            ))
        
        # Agent 数据：agent_position_item.shape 为 [A, P, 2]，agent_mask_item.shape 为 [A, P]
        A = agent_position_item.shape[0]
        for agent_idx in range(max_A):
            # 仅当 agent_idx 等于 select_agent_id 且数据存在时显示
            if agent_idx < A and agent_idx == select_agent_id:
                agent_line = agent_position_item[agent_idx]  # shape: [P, 2]
                agent_mask = agent_mask_item[agent_idx]        # shape: [P]
                valid_points = agent_line[agent_mask]
            else:
                valid_points = torch.empty((0, 2))
            
            traces.append(go.Scatter(
                x=valid_points[:, 0].tolist() if valid_points.shape[0] > 0 else [],
                y=valid_points[:, 1].tolist() if valid_points.shape[0] > 0 else [],
                mode='lines+markers',
                name=f"Agent {agent_idx}",
                line=dict(color=agent_colors[agent_idx])
            ))
        
        frames.append(dict(data=traces, name=str(frame_idx)))
    
    # 初始帧数据采用第 0 帧（所有帧中 trace 数量和顺序均一致）
    initial_data = frames[0]['data'] if frames else []
    
    # 根据第一帧数据，确定车道 trace 数量（假设每帧车道数量一致）
    initial_D1 = map_data[0].shape[0]
    lane_trace_count = initial_D1 * 3   # 每个车道 3 个 trace
    agent_trace_count = max_A           # agent 部分 trace 数量
    
    # 计算各组 trace 索引（用于后续下拉菜单控制）
    lane_indices = list(range(lane_trace_count))
    agent_indices = list(range(lane_trace_count, lane_trace_count + agent_trace_count))
    
    # 构造 Figure 对象，添加播放/暂停按钮、帧滑动条，以及下拉菜单（此处保留车道/agent显示控制，可根据需要修改）
    fig = go.Figure(
        data=initial_data,
        layout=go.Layout(
            title=f"Map & Agent Data (Selected Agent {select_agent_id})",
            xaxis=dict(title="X"),
            yaxis=dict(title="Y", scaleanchor="x", scaleratio=1),
            template="plotly_white",
            width=800,
            height=800,
            uirevision="constant",  # 保持轴状态不刷新
            updatemenus=[
                # 播放/暂停按钮（放在图下中间偏左）
                {
                    "type": "buttons",
                    "showactive": False,
                    "x": 0,
                    "y": -0.3,
                    "xanchor": "left",
                    "yanchor": "top",
                    "pad": {"t": 10, "r": 10},
                    "buttons": [
                        {
                            "label": "播放",
                            "method": "animate",
                            "args": [None, {"frame": {"duration": 0, "redraw": True},
                                            "fromcurrent": True, "transition": {"duration": 0}}]
                        },
                        {
                            "label": "暂停",
                            "method": "animate",
                            "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                              "mode": "immediate",
                                              "transition": {"duration": 0}}]
                        }
                    ]
                },
                # 下拉菜单：独立控制车道显示（放在图下左侧）
                {
                    "type": "dropdown",
                    "direction": "down",
                    "showactive": True,
                    "x": 0,
                    "y": -0.1,
                    "xanchor": "left",
                    "yanchor": "top",
                    "pad": {"t": 10, "r": 10},
                    "buttons": [
                        {
                            "label": "显示车道",
                            "method": "restyle",
                            "args": [{"visible": True}, lane_indices]
                        },
                        {
                            "label": "隐藏车道",
                            "method": "restyle",
                            "args": [{"visible": False}, lane_indices]
                        }
                    ]
                },
                # 下拉菜单：独立控制 agent 显示（放在图下右侧）
                {
                    "type": "dropdown",
                    "direction": "down",
                    "showactive": True,
                    "x": 1,
                    "y": -0.1,
                    "xanchor": "right",
                    "yanchor": "top",
                    "pad": {"t": 10, "r": 10},
                    "buttons": [
                        {
                            "label": "显示 Agent",
                            "method": "restyle",
                            "args": [{"visible": True}, agent_indices]
                        },
                        {
                            "label": "隐藏 Agent",
                            "method": "restyle",
                            "args": [{"visible": False}, agent_indices]
                        }
                    ]
                }
            ],
            # 滑动条放在图下中间
            sliders=[{
                "active": 0,
                "x": 0.5,
                "y": -0.5,
                "xanchor": "center",
                "yanchor": "top",
                "currentvalue": {
                    "font": {"size": 20},
                    "prefix": "Frame: ",
                    "visible": True,
                    "xanchor": "right"
                },
                "pad": {"b": 10, "t": 50},
                "len": 0.9,
                "steps": [{
                    "args": [[str(k)],
                             {"frame": {"duration": 0, "redraw": True},
                              "mode": "immediate",
                              "transition": {"duration": 0}}],
                    "label": str(k),
                    "method": "animate"
                } for k in range(n_frames)]
            }]
        ),
        frames=frames
    )
    return fig

# -------------------- 交互界面 --------------------
# 使用 ipywidgets 创建一个整数滑块来选择 agent id（取值范围：0 到 max_A-1）
agent_slider = widgets.IntSlider(min=0, max=max_A-1, step=1, value=1, description="Agent ID:")
output = widgets.Output()

def update_fig(agent_id):
    with output:
        output.clear_output(wait=True)
        fig = build_figure(agent_id)
        fig.show()

# 显示交互控件
widgets.interact(update_fig, agent_id=agent_slider)
display(output)
import torch
import plotly.graph_objects as go
from plotly.colors import qualitative
import ipywidgets as widgets
from IPython.display import display

# -------------------- 数据加载 --------------------
# 加载保存的 pt 文件
map_data = torch.load("./map_data.pt")
map_mask_data = torch.load("./map_mask_data.pt")
agent_position_data = torch.load("./agent_position_data.pt")
agent_mask_data = torch.load("./agent_mask_data.pt")

# -------------------- 参数设置 --------------------
# 定义地图车道颜色组，每组包含中心线、左边界和右边界的颜色
color_pairs = [
    {"center": "#2C3E50", "left": "#1f77b4", "right": "#d62728"},  # 组0：中心线带蓝色调
    {"center": "#34495E", "left": "#17becf", "right": "#ff7f0e"},  # 组1：中心线带蓝灰调
    {"center": "#8E44AD", "left": "#0074D9", "right": "#FF4136"},  # 组2：中心线带紫色调
    {"center": "#16A085", "left": "#2E86C1", "right": "#E74C3C"},  # 组3：中心线带青绿色调
    {"center": "#27AE60", "left": "#2980B9", "right": "#E67E22"},  # 组4：中心线带绿色调
    {"center": "#D35400", "left": "#3498DB", "right": "#F39C12"},  # 组5：中心线带橙色调
    {"center": "#C0392B", "left": "#5DADE2", "right": "#E74C3C"},  # 组6：中心线带红色调
]

# 计算所有帧中 agent 数量的最大值（agent 按 A 维度排序，即相同位置即为同一目标）
max_A = max([agent_position_item.shape[0] for agent_position_item in agent_position_data])
# 使用 Plotly 的定性配色方案作为 agent 的颜色池（不足时循环使用）
agent_colors_pool = qualitative.Plotly  
agent_colors = [agent_colors_pool[i % len(agent_colors_pool)] for i in range(max_A)]
n_frames = len(map_data)

# -------------------- 构建动画图的函数 --------------------
def build_figure(select_agent_id):
    """
    根据所选 agent id（select_agent_id）构建动画图，
    仅显示该 agent（位置为 select_agent_id），其它 agent 添加空 trace，
    保证每一帧中 agent trace 的顺序和数量一致，从而使得颜色稳定。
    """
    frames = []
    
    for frame_idx, (map_item, map_mask_item, agent_position_item, agent_mask_item) in enumerate(
            zip(map_data, map_mask_data, agent_position_data, agent_mask_data)):
        
        traces = []
        # 地图数据：假设 map_item.shape 为 [D1, D2, P, 2]，其中 D1 为车道数量
        D1, _, _, _ = map_item.shape
        for lane_idx in range(D1):
            color_set = color_pairs[lane_idx % len(color_pairs)]
            lane_mask = map_mask_item[lane_idx]  # 布尔 mask, shape: [P]
            lines = map_item[lane_idx]           # shape: [D2, P, 2]
            center_line = lines[0]
            left_bound = lines[1]
            right_bound = lines[2]
            # 筛选有效点
            center_valid = center_line[lane_mask]
            left_valid = left_bound[lane_mask]
            right_valid = right_bound[lane_mask]
            
            # 始终添加三个 trace（数据为空时 x,y 为 []）
            traces.append(go.Scatter(
                x=center_valid[:, 0].tolist() if center_valid.shape[0] > 0 else [],
                y=center_valid[:, 1].tolist() if center_valid.shape[0] > 0 else [],
                mode='lines',
                name=f"Lane {lane_idx} 中心线",
                line=dict(color=color_set["center"], dash="dot")
            ))
            traces.append(go.Scatter(
                x=left_valid[:, 0].tolist() if left_valid.shape[0] > 0 else [],
                y=left_valid[:, 1].tolist() if left_valid.shape[0] > 0 else [],
                mode='lines',
                name=f"Lane {lane_idx} 左边界",
                line=dict(color=color_set["left"])
            ))
            traces.append(go.Scatter(
                x=right_valid[:, 0].tolist() if right_valid.shape[0] > 0 else [],
                y=right_valid[:, 1].tolist() if right_valid.shape[0] > 0 else [],
                mode='lines',
                name=f"Lane {lane_idx} 右边界",
                line=dict(color=color_set["right"])
            ))
        
        # Agent 数据：agent_position_item.shape 为 [A, P, 2]，agent_mask_item.shape 为 [A, P]
        A = agent_position_item.shape[0]
        for agent_idx in range(max_A):
            # 仅当 agent_idx 等于 select_agent_id 且数据存在时显示
            if agent_idx < A and agent_idx == select_agent_id:
                agent_line = agent_position_item[agent_idx]  # shape: [P, 2]
                agent_mask = agent_mask_item[agent_idx]        # shape: [P]
                valid_points = agent_line[agent_mask]
            else:
                valid_points = torch.empty((0, 2))
            
            traces.append(go.Scatter(
                x=valid_points[:, 0].tolist() if valid_points.shape[0] > 0 else [],
                y=valid_points[:, 1].tolist() if valid_points.shape[0] > 0 else [],
                mode='lines+markers',
                name=f"Agent {agent_idx}",
                line=dict(color=agent_colors[agent_idx])
            ))
        
        frames.append(dict(data=traces, name=str(frame_idx)))
    
    # 初始帧数据采用第 0 帧（所有帧中 trace 数量和顺序均一致）
    initial_data = frames[0]['data'] if frames else []
    
    # 根据第一帧数据，确定车道 trace 数量（假设每帧车道数量一致）
    initial_D1 = map_data[0].shape[0]
    lane_trace_count = initial_D1 * 3   # 每个车道 3 个 trace
    agent_trace_count = max_A           # agent 部分 trace 数量
    
    # 计算各组 trace 索引（用于后续下拉菜单控制）
    lane_indices = list(range(lane_trace_count))
    agent_indices = list(range(lane_trace_count, lane_trace_count + agent_trace_count))
    
    # 构造 Figure 对象，添加播放/暂停按钮、帧滑动条，以及下拉菜单（此处保留车道/agent显示控制，可根据需要修改）
    fig = go.Figure(
        data=initial_data,
        layout=go.Layout(
            title=f"Map & Agent Data (Selected Agent {select_agent_id})",
            xaxis=dict(title="X"),
            yaxis=dict(title="Y", scaleanchor="x", scaleratio=1),
            template="plotly_white",
            width=800,
            height=800,
            uirevision="constant",  # 保持轴状态不刷新
            updatemenus=[
                # 播放/暂停按钮（放在图下中间偏左）
                {
                    "type": "buttons",
                    "showactive": False,
                    "x": 0,
                    "y": -0.3,
                    "xanchor": "left",
                    "yanchor": "top",
                    "pad": {"t": 10, "r": 10},
                    "buttons": [
                        {
                            "label": "播放",
                            "method": "animate",
                            "args": [None, {"frame": {"duration": 0, "redraw": True},
                                            "fromcurrent": True, "transition": {"duration": 0}}]
                        },
                        {
                            "label": "暂停",
                            "method": "animate",
                            "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                              "mode": "immediate",
                                              "transition": {"duration": 0}}]
                        }
                    ]
                },
                # 下拉菜单：独立控制车道显示（放在图下左侧）
                {
                    "type": "dropdown",
                    "direction": "down",
                    "showactive": True,
                    "x": 0,
                    "y": -0.1,
                    "xanchor": "left",
                    "yanchor": "top",
                    "pad": {"t": 10, "r": 10},
                    "buttons": [
                        {
                            "label": "显示车道",
                            "method": "restyle",
                            "args": [{"visible": True}, lane_indices]
                        },
                        {
                            "label": "隐藏车道",
                            "method": "restyle",
                            "args": [{"visible": False}, lane_indices]
                        }
                    ]
                },
                # 下拉菜单：独立控制 agent 显示（放在图下右侧）
                {
                    "type": "dropdown",
                    "direction": "down",
                    "showactive": True,
                    "x": 1,
                    "y": -0.1,
                    "xanchor": "right",
                    "yanchor": "top",
                    "pad": {"t": 10, "r": 10},
                    "buttons": [
                        {
                            "label": "显示 Agent",
                            "method": "restyle",
                            "args": [{"visible": True}, agent_indices]
                        },
                        {
                            "label": "隐藏 Agent",
                            "method": "restyle",
                            "args": [{"visible": False}, agent_indices]
                        }
                    ]
                }
            ],
            # 滑动条放在图下中间
            sliders=[{
                "active": 0,
                "x": 0.5,
                "y": -0.5,
                "xanchor": "center",
                "yanchor": "top",
                "currentvalue": {
                    "font": {"size": 20},
                    "prefix": "Frame: ",
                    "visible": True,
                    "xanchor": "right"
                },
                "pad": {"b": 10, "t": 50},
                "len": 0.9,
                "steps": [{
                    "args": [[str(k)],
                             {"frame": {"duration": 0, "redraw": True},
                              "mode": "immediate",
                              "transition": {"duration": 0}}],
                    "label": str(k),
                    "method": "animate"
                } for k in range(n_frames)]
            }]
        ),
        frames=frames
    )
    return fig

# -------------------- 交互界面 --------------------
# 使用 ipywidgets 创建一个整数滑块来选择 agent id（取值范围：0 到 max_A-1）
agent_slider = widgets.IntSlider(min=0, max=max_A-1, step=1, value=1, description="Agent ID:")
output = widgets.Output()

def update_fig(agent_id):
    with output:
        output.clear_output(wait=True)
        fig = build_figure(agent_id)
        fig.show()

# 显示交互控件
widgets.interact(update_fig, agent_id=agent_slider)
display(output)
import torch
import plotly.graph_objects as go
from plotly.colors import qualitative
import ipywidgets as widgets
from IPython.display import display

# -------------------- 数据加载 --------------------
# 加载保存的 pt 文件
map_data = torch.load("./map_data.pt")
map_mask_data = torch.load("./map_mask_data.pt")
agent_position_data = torch.load("./agent_position_data.pt")
agent_mask_data = torch.load("./agent_mask_data.pt")

# -------------------- 参数设置 --------------------
# 定义地图车道颜色组，每组包含中心线、左边界和右边界的颜色
color_pairs = [
    {"center": "#2C3E50", "left": "#1f77b4", "right": "#d62728"},  # 组0：中心线带蓝色调
    {"center": "#34495E", "left": "#17becf", "right": "#ff7f0e"},  # 组1：中心线带蓝灰调
    {"center": "#8E44AD", "left": "#0074D9", "right": "#FF4136"},  # 组2：中心线带紫色调
    {"center": "#16A085", "left": "#2E86C1", "right": "#E74C3C"},  # 组3：中心线带青绿色调
    {"center": "#27AE60", "left": "#2980B9", "right": "#E67E22"},  # 组4：中心线带绿色调
    {"center": "#D35400", "left": "#3498DB", "right": "#F39C12"},  # 组5：中心线带橙色调
    {"center": "#C0392B", "left": "#5DADE2", "right": "#E74C3C"},  # 组6：中心线带红色调
]

# 计算所有帧中 agent 数量的最大值（agent 按 A 维度排序，即相同位置即为同一目标）
max_A = max([agent_position_item.shape[0] for agent_position_item in agent_position_data])
# 使用 Plotly 的定性配色方案作为 agent 的颜色池（不足时循环使用）
agent_colors_pool = qualitative.Plotly  
agent_colors = [agent_colors_pool[i % len(agent_colors_pool)] for i in range(max_A)]
n_frames = len(map_data)

# -------------------- 构建动画图的函数 --------------------
def build_figure(select_agent_id):
    """
    根据所选 agent id（select_agent_id）构建动画图，
    仅显示该 agent（位置为 select_agent_id），其它 agent 添加空 trace，
    保证每一帧中 agent trace 的顺序和数量一致，从而使得颜色稳定。
    """
    frames = []
    
    for frame_idx, (map_item, map_mask_item, agent_position_item, agent_mask_item) in enumerate(
            zip(map_data, map_mask_data, agent_position_data, agent_mask_data)):
        
        traces = []
        # 地图数据：假设 map_item.shape 为 [D1, D2, P, 2]，其中 D1 为车道数量
        D1, _, _, _ = map_item.shape
        for lane_idx in range(D1):
            color_set = color_pairs[lane_idx % len(color_pairs)]
            lane_mask = map_mask_item[lane_idx]  # 布尔 mask, shape: [P]
            lines = map_item[lane_idx]           # shape: [D2, P, 2]
            center_line = lines[0]
            left_bound = lines[1]
            right_bound = lines[2]
            # 筛选有效点
            center_valid = center_line[lane_mask]
            left_valid = left_bound[lane_mask]
            right_valid = right_bound[lane_mask]
            
            # 始终添加三个 trace（数据为空时 x,y 为 []）
            traces.append(go.Scatter(
                x=center_valid[:, 0].tolist() if center_valid.shape[0] > 0 else [],
                y=center_valid[:, 1].tolist() if center_valid.shape[0] > 0 else [],
                mode='lines',
                name=f"Lane {lane_idx} 中心线",
                line=dict(color=color_set["center"], dash="dot")
            ))
            traces.append(go.Scatter(
                x=left_valid[:, 0].tolist() if left_valid.shape[0] > 0 else [],
                y=left_valid[:, 1].tolist() if left_valid.shape[0] > 0 else [],
                mode='lines',
                name=f"Lane {lane_idx} 左边界",
                line=dict(color=color_set["left"])
            ))
            traces.append(go.Scatter(
                x=right_valid[:, 0].tolist() if right_valid.shape[0] > 0 else [],
                y=right_valid[:, 1].tolist() if right_valid.shape[0] > 0 else [],
                mode='lines',
                name=f"Lane {lane_idx} 右边界",
                line=dict(color=color_set["right"])
            ))
        
        # Agent 数据：agent_position_item.shape 为 [A, P, 2]，agent_mask_item.shape 为 [A, P]
        A = agent_position_item.shape[0]
        for agent_idx in range(max_A):
            # 仅当 agent_idx 等于 select_agent_id 且数据存在时显示
            if agent_idx < A and agent_idx == select_agent_id:
                agent_line = agent_position_item[agent_idx]  # shape: [P, 2]
                agent_mask = agent_mask_item[agent_idx]        # shape: [P]
                valid_points = agent_line[agent_mask]
            else:
                valid_points = torch.empty((0, 2))
            
            traces.append(go.Scatter(
                x=valid_points[:, 0].tolist() if valid_points.shape[0] > 0 else [],
                y=valid_points[:, 1].tolist() if valid_points.shape[0] > 0 else [],
                mode='lines+markers',
                name=f"Agent {agent_idx}",
                line=dict(color=agent_colors[agent_idx])
            ))
        
        frames.append(dict(data=traces, name=str(frame_idx)))
    
    # 初始帧数据采用第 0 帧（所有帧中 trace 数量和顺序均一致）
    initial_data = frames[0]['data'] if frames else []
    
    # 根据第一帧数据，确定车道 trace 数量（假设每帧车道数量一致）
    initial_D1 = map_data[0].shape[0]
    lane_trace_count = initial_D1 * 3   # 每个车道 3 个 trace
    agent_trace_count = max_A           # agent 部分 trace 数量
    
    # 计算各组 trace 索引（用于后续下拉菜单控制）
    lane_indices = list(range(lane_trace_count))
    agent_indices = list(range(lane_trace_count, lane_trace_count + agent_trace_count))
    
    # 构造 Figure 对象，添加播放/暂停按钮、帧滑动条，以及下拉菜单（此处保留车道/agent显示控制，可根据需要修改）
    fig = go.Figure(
        data=initial_data,
        layout=go.Layout(
            title=f"Map & Agent Data (Selected Agent {select_agent_id})",
            xaxis=dict(title="X"),
            yaxis=dict(title="Y", scaleanchor="x", scaleratio=1),
            template="plotly_white",
            width=800,
            height=800,
            uirevision="constant",  # 保持轴状态不刷新
            updatemenus=[
                # 播放/暂停按钮（放在图下中间偏左）
                {
                    "type": "buttons",
                    "showactive": False,
                    "x": 0,
                    "y": -0.3,
                    "xanchor": "left",
                    "yanchor": "top",
                    "pad": {"t": 10, "r": 10},
                    "buttons": [
                        {
                            "label": "播放",
                            "method": "animate",
                            "args": [None, {"frame": {"duration": 0, "redraw": True},
                                            "fromcurrent": True, "transition": {"duration": 0}}]
                        },
                        {
                            "label": "暂停",
                            "method": "animate",
                            "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                              "mode": "immediate",
                                              "transition": {"duration": 0}}]
                        }
                    ]
                },
                # 下拉菜单：独立控制车道显示（放在图下左侧）
                {
                    "type": "dropdown",
                    "direction": "down",
                    "showactive": True,
                    "x": 0,
                    "y": -0.1,
                    "xanchor": "left",
                    "yanchor": "top",
                    "pad": {"t": 10, "r": 10},
                    "buttons": [
                        {
                            "label": "显示车道",
                            "method": "restyle",
                            "args": [{"visible": True}, lane_indices]
                        },
                        {
                            "label": "隐藏车道",
                            "method": "restyle",
                            "args": [{"visible": False}, lane_indices]
                        }
                    ]
                },
                # 下拉菜单：独立控制 agent 显示（放在图下右侧）
                {
                    "type": "dropdown",
                    "direction": "down",
                    "showactive": True,
                    "x": 1,
                    "y": -0.1,
                    "xanchor": "right",
                    "yanchor": "top",
                    "pad": {"t": 10, "r": 10},
                    "buttons": [
                        {
                            "label": "显示 Agent",
                            "method": "restyle",
                            "args": [{"visible": True}, agent_indices]
                        },
                        {
                            "label": "隐藏 Agent",
                            "method": "restyle",
                            "args": [{"visible": False}, agent_indices]
                        }
                    ]
                }
            ],
            # 滑动条放在图下中间
            sliders=[{
                "active": 0,
                "x": 0.5,
                "y": -0.5,
                "xanchor": "center",
                "yanchor": "top",
                "currentvalue": {
                    "font": {"size": 20},
                    "prefix": "Frame: ",
                    "visible": True,
                    "xanchor": "right"
                },
                "pad": {"b": 10, "t": 50},
                "len": 0.9,
                "steps": [{
                    "args": [[str(k)],
                             {"frame": {"duration": 0, "redraw": True},
                              "mode": "immediate",
                              "transition": {"duration": 0}}],
                    "label": str(k),
                    "method": "animate"
                } for k in range(n_frames)]
            }]
        ),
        frames=frames
    )
    return fig

# -------------------- 交互界面 --------------------
# 使用 ipywidgets 创建一个整数滑块来选择 agent id（取值范围：0 到 max_A-1）
agent_slider = widgets.IntSlider(min=0, max=max_A-1, step=1, value=1, description="Agent ID:")
output = widgets.Output()

def update_fig(agent_id):
    with output:
        output.clear_output(wait=True)
        fig = build_figure(agent_id)
        fig.show()

# 显示交互控件
widgets.interact(update_fig, agent_id=agent_slider)
display(output)

## TL;NR
Based on PLUTO, we study the integrating method of long and short-term decision making, and the Time Dependent Normalization achieves the most significant improvement to 91.32% in the nuPlan Val4 CLS-NR score.

## Performance Comparison with SOTA methods
w/o post-processing
|  Model Name  | Val14 CLS-NR Score  | Val14 CLS-R Score  |
|  ----  | ----  | ----  |
| [PLUTO](https://github.com/jchengai/pluto)  | 89.04 | 80.01 |
| [STR2-CPKS-800M](https://github.com/Tsinghua-MARS-Lab/StateTransformer?tab=readme-ov-file)| 65.16 | - |
| [Diffusion Planner](https://github.com/ZhengYinan-AIR/Diffusion-Planner)  | 89.87 | **82.80** |
| PlanScope (ours)  | **91.32** | 80.96 |

<!-- Hybrid Mode
|  Model Name  | Val14 CLS-NR Score  | Val14 CLS-R Score  |
|  ----  | ----  | ----  |
| [PLUTO](https://github.com/jchengai/pluto)  | 93.21 | 92.06 |
| [STR2-CPKS-800M](https://github.com/Tsinghua-MARS-Lab/StateTransformer?tab=readme-ov-file)| 93.91 | 92.51 |
| [Diffusion Planner](https://github.com/ZhengYinan-AIR/Diffusion-Planner)  | **94.26** | **92.90** |
| PlanScope (ours)  | 93.59 | 91.07 | -->

## Setup Environment

### Setup dataset

Setup the nuPlan dataset following the [official-doc](https://nuplan-devkit.readthedocs.io/en/latest/dataset_setup.html)

### Setup conda environment

```
conda create -n planscope python=3.9
conda activate planscope

# install nuplan-devkit
git clone https://github.com/motional/nuplan-devkit.git && cd nuplan-devkit
pip install -e .
pip install -r ./requirements.txt

# setup planscope
cd ..
git clone https://github.com/Rex-sys-hk/PlanScope && cd planscope
sh ./script/setup_env.sh
```

## Feature Cache

Preprocess the dataset to accelerate training. It is recommended to run a small sanity check to make sure everything is correctly setup.

```
 python run_training.py \
    py_func=cache +training=train_pluto \
    scenario_builder=nuplan_mini \
    cache.cache_path=/nuplan/exp/sanity_check \
    cache.cleanup_cache=true \
    scenario_filter=training_scenarios_tiny \
    worker=sequential
```

Then preprocess the whole nuPlan training set (this will take some time). You may need to change `cache.cache_path` to suit your condition

```
 export PYTHONPATH=$PYTHONPATH:$(pwd)

 python run_training.py \
    py_func=cache +training=train_pluto \
    scenario_builder=nuplan \
    cache.cache_path=/nuplan/exp/cache_pluto_1M \
    cache.cleanup_cache=true \
    scenario_filter=training_scenarios_1M \
    worker.threads_per_node=40
```

## Training


```
sh train_scope.sh
```

- you can remove wandb related configurations if your prefer tensorboard.


## Checkpoint

Copy your chekpoint path to ```sim_scope.sh``` or ```sim_pluto.sh``` and replace the value of ```CKPT_N``` to run the evaluation. 

<!-- | PlanScope-h10-m6    | [OneDrive](https://hkustconnect-my.sharepoint.com/:u:/g/personal/rxin_connect_ust_hk/EcxJsqO4QgxJt2HeyfmDEssBelkGmMqzq3pFkk2w5OgQDQ?e=bUem3P)|
| PlanScope-h20-m6    | [OneDrive](https://hkustconnect-my.sharepoint.com/:u:/g/personal/rxin_connect_ust_hk/EbdjCkpdTEBKhwnz4VFv0R8BDD0C76zHsV7BedgYlytV5g?e=9BA7ft)| -->

| Model            | Download |
| ---------------- | -------- |
| Pluto-aux-nocil-m6-baseline  | [OneDrive](https://hkustconnect-my.sharepoint.com/:u:/g/personal/rxin_connect_ust_hk/EYkVd-OcOTFLlP5KE7ZnG-0BrluObe4vd7jNAhHeKtmcjw?e=UBmqf1)|
| PlanScope-Ih10-DWT | [OneDrive](https://hkustconnect-my.sharepoint.com/:u:/g/personal/rxin_connect_ust_hk/EXjVIgwKh3hCmMfJ-rQArcABRn3tH1RZhptPOLYRJjkS2A?e=scYt4e)    |
| PlanScope-Mh10-DWH | [OneDrive](https://hkustconnect-my.sharepoint.com/:u:/g/personal/rxin_connect_ust_hk/EXVaD_lc3kJBtUxGSQBBgPwBl8isEQzRaDtfrJ-geDB-XQ?e=pnbSPy)    |
| PlanScope-Mh20-DWT | [OneDrive](https://hkustconnect-my.sharepoint.com/:u:/g/personal/rxin_connect_ust_hk/EajN1DzBjKhMg4GiqkuuHuoBGilZzJbkK5QiPD9_GuoDLQ?e=BgidZM)    |
| --- |
| PlanScope-Th20 | [OneDrive](https://hkustconnect-my.sharepoint.com/:u:/g/personal/rxin_connect_ust_hk/EcHd8CFgBH1JqKT9yMyPsr0BukUsXTjfJpNSik_vQQrsLw?e=48VbzA)    |
| PlanScope-timedecay | [OneDrive](https://hkustconnect-my.sharepoint.com/:u:/g/personal/rxin_connect_ust_hk/EdMfIvFKuFlLh-SyHVvMB74Bs3TxH5hEp3HCSU34b6yAjg?e=KmVDGh)    |
| PlanScope-timenorm | [OneDrive](https://hkustconnect-my.sharepoint.com/:u:/g/personal/rxin_connect_ust_hk/EUMawRA-i-NIimhVp_I_Ft8BeuHWrCJzsVXb-E4BEMMQuA?e=0uRrDN)    |
| --- |
| Pluto-1M-aux-cil-m12-original | [OneDrive](https://hkustconnect-my.sharepoint.com/:u:/g/personal/jchengai_connect_ust_hk/EaFpLwwHFYVKsPVLH2nW5nEBNbPS7gqqu_Rv2V1dzODO-Q?e=LAZQcI)    |
| PlanScope-timenorm-cil-m12 | [OneDrive](https://hkustconnect-my.sharepoint.com/:u:/g/personal/rxin_connect_ust_hk/Ed863-9h9ZtFm145JyWGjCIBbF-rInj8P2smuXeG0SAPsg?e=g860Ho)    |

## Run PlanScope-planner simulation

Run simulation for a random scenario in the nuPlan-mini split

```
sh ./sim_scope.sh
```


## To Do

The code is under cleaning and will be released gradually.

- [ ] improve docs
- [x] training code
- [x] visualization
- [x] Scope-planner & checkpoint
- [x] feature builder & model
- [x] initial repo & paper

## Citation

If you find this repo useful, please consider giving us a star 🌟 and citing our related paper.

```bibtex
@misc{planscope,
      title={{PlanScope:} Learning to Plan Within Decision Scope Does Matter}, 
      author={Ren Xin and Jie Cheng and Jun Ma},
      year={2024},
      eprint={2411.00476},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2411.00476}, 
}
```

## Thanks
- [tuplan](https://github.com/autonomousvision/tuplan_garage)
- [pluto](https://github.com/jchengai/pluto)


<!-- ## Special Announcement (Updated on 4 March 2025)

Our approach has achieved a CLS-NR score of 91.32% without rule-based post-processing, which currently is the highest score in pure-model-mode. 
However, the main objective is to find a general method for addressing horizon fusing problem, thus enhance the performance of planning models during execution. -->

<!-- This work investigates a technique to enhance the performance of planning models in a pure learning framework. We have deliberately omitted the rule-based pre- and post-processing modules from the baseline approach to mitigate the impact of artificially crafted rules, as claimed in our paper. A certain unauthorized publication led to **inaccuracies in the depiction of its state-of-the-art (SOTA) capabilities**. We hereby clarify this to prevent misunderstanding.

Nevertheless, the method introduced in our article is worth trying and could potentially serve as an add-on to augment the performance of the models you are developing, especially when the dataset is small. We are open to sharing and discussing evaluation results to foster a collaborative exchange. -->

## Others
- Please mind the common problem of nuPlan Dataset setup: https://github.com/motional/nuplan-devkit/issues/379 
- Advised NATTEN Version: 0.14.6+torch1121cu116
- Please mind your Linux system version, Ubuntu 18.04.6 LTS is prefered. Debian may lead to some unexpected error in closed-loop simulation.
- When training on the 20% dataset, the random selection of data splits during training possibly cause fluctuations of about 2% CLS-NR score on Random14, the training on partial dataset should only be used as reference during development.
- This repo is updated on 5 March 2025, the previous version can be found by checkout branch archived_1.
