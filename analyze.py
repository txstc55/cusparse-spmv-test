f = open("performance.txt", "r")
times = []
for line in f:
    if "executions" in line:
        times.append(float(line.split(":")[-1].split(" ")[1]) / 1000)

matrix_size = [5000 * (2 ** i) for i in range(int(len(times) / 7))]
print(matrix_size)
times1 = times[::7]
times2 = times[1::7]
times3 = times[2::7]
times4 = times[3::7]
times5 = times[4::7]
times6 = times[5::7]
times7 = times[6::7]

names = ["cuSPARSE CSR SpMV", "cuSPARSE COO SpMV", "Symmetric BCSR SpMV", "Symmetric BCOO SpMV", "Symmetric BCOO Backward SpMV", "Symmetric BCOO FCFS SpMV", "Symmetric BCOO FCFS Backward SpMV"]

from matplotlib import pyplot as plt

import plotly.graph_objects as go

# Assuming the rest of your code is above this point

# Create a Plotly figure
fig = go.Figure()

# Add each line plot
fig.add_trace(go.Scatter(x=matrix_size, y=times1, mode='lines+markers', name=names[0], marker_symbol='x'))
fig.add_trace(go.Scatter(x=matrix_size, y=times2, mode='lines+markers', name=names[1], marker_symbol='x'))
fig.add_trace(go.Scatter(x=matrix_size, y=times3, mode='lines+markers', name=names[2], marker_symbol='x'))
fig.add_trace(go.Scatter(x=matrix_size, y=times4, mode='lines+markers', name=names[3], marker_symbol='x'))
fig.add_trace(go.Scatter(x=matrix_size, y=times5, mode='lines+markers', name=names[4], marker_symbol='x'))
fig.add_trace(go.Scatter(x=matrix_size, y=times6, mode='lines+markers', name=names[5], marker_symbol='x'))
fig.add_trace(go.Scatter(x=matrix_size, y=times7, mode='lines+markers', name=names[6], marker_symbol='x'))


# Update layout for log scale and add titles
fig.update_layout(
    title="SpMV Performance Comparison",
    xaxis=dict(type='log', title='Vertex Count'),
    yaxis=dict(type='log', title='Time (ms)'),
    legend_title="Method"
)

# Show the interactive plot
fig.show()

# Optionally, save the plot as an HTML file
fig.write_html("interactive_performance_plot.html")