f = open("performance.txt", "r")
times = []
for line in f:
    if "executions" in line:
        times.append(float(line.split(":")[-1].split(" ")[1]) / 1000)

matrix_size = [5000 * (2 ** i) for i in range(int(len(times) / 4))]
print(matrix_size)
times1 = times[::4]
times2 = times[1::4]
times3 = times[2::4]
times4 = times[3::4]

names = ["cuSPARSE CSR SpMV", "cuSPARSE COO SpMV", "Symmetric BCSR SpMV", "Symmetric BCOO SpMV"]

from matplotlib import pyplot as plt

# fig, ax = plt.subplots()
# ax.set_ylabel("Time (ms)")
# ax.set_xlabel("Vertex Count")
# ax.set_title("SpMV Performance Comparison")
# ax.plot(matrix_size, times1, label=names[0], marker="o")
# ax.plot(matrix_size, times2, label=names[1])
# ax.plot(matrix_size, times3, label=names[2])
# ax.plot(matrix_size, times4, label=names[3], marker="x")
# ax.legend()

# ## set the x axis to be log scale
# ax.set_xscale('log')
# plt.xticks(ticks = matrix_size, labels=matrix_size)


# ax.set_yscale('log')

import plotly.graph_objects as go

# Assuming the rest of your code is above this point

# Create a Plotly figure
fig = go.Figure()

# Add each line plot
fig.add_trace(go.Scatter(x=matrix_size, y=times1, mode='lines+markers', name=names[0]))
fig.add_trace(go.Scatter(x=matrix_size, y=times2, mode='lines', name=names[1]))
fig.add_trace(go.Scatter(x=matrix_size, y=times3, mode='lines', name=names[2]))
fig.add_trace(go.Scatter(x=matrix_size, y=times4, mode='lines+markers', name=names[3], marker_symbol='x'))

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